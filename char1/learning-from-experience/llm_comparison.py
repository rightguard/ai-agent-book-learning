"""实验 8-2：Q-learning 与 Qwen Agent 寻宝对照（完全独立单文件）。

无需 game_environment.py、rl_agent.py、llm_agent.py 或任何其他项目 Python 文件。
同一确定性寻宝规则：Q-learning 训练 10000 局、评估 100 局；Qwen 首次挑战 1 局。
模型通过百炼 Chat Completions 逐步选择游戏动作；保留完整请求和行动证据。
依赖：pip install openai python-dotenv pyyaml tqdm matplotlib
运行：python char1/learning-from-experience/llm_comparison.py
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# 1. 寻宝游戏环境：内嵌原项目实验 8-1 的确定性规则
# ============================================================
class ItemType(Enum):
    KEY = "key"
    WEAPON = "weapon"
    TREASURE = "treasure"
    TOOL = "tool"
    POTION = "potion"


@dataclass
class Item:
    name: str
    item_type: ItemType
    description: str
    properties: dict = field(default_factory=dict)


@dataclass
class Room:
    name: str
    description: str
    items: List[Item] = field(default_factory=list)
    exits: Dict[str, str] = field(default_factory=dict)
    locked_exits: Dict[str, str] = field(default_factory=dict)
    has_guard: bool = False
    guard_defeated: bool = False


class TreasureHuntGame:
    """5 房间文字寻宝。默认规则与官方确定性模式一致，至多 50 步。"""

    def __init__(self, seed: Optional[int] = None):
        self.rooms = {}
        self.current_room = None
        self.inventory = []
        self.score = 0
        self.moves = 0
        self.max_moves = 50
        self.game_over = False
        self.victory = False
        self.active_effects = {}

        # 游戏中的隐藏规则：Q-learning 不会提前得到这些映射。
        self.color_key_mapping = {
            "red key": "red door",
            "blue key": "blue door",
            "golden key": "golden door",
        }
        self.weapon_effectiveness = {
            "rusty sword": ["weak guard"],
            "silver sword": ["weak guard", "strong guard", "dragon"],
        }
        self.crafting_recipes = {
            frozenset(["rusty sword", "magic crystal"]): "silver sword"
        }
        self._initialize_world()

    def _initialize_world(self):
        self.rooms["entrance"] = Room(
            name="entrance",
            description="You stand in a dimly lit entrance hall. Stone walls echo your footsteps.",
            items=[Item("rusty sword", ItemType.WEAPON, "An old sword with rust spots")],
            exits={"north": "hallway", "east": "storage"},
        )
        self.rooms["storage"] = Room(
            name="storage",
            description="A dusty storage room filled with old crates and barrels.",
            items=[
                Item("red key", ItemType.KEY, "A small red metal key"),
                Item("magic crystal", ItemType.TOOL, "A glowing crystal that hums with energy"),
            ],
            exits={"west": "entrance"},
        )
        self.rooms["hallway"] = Room(
            name="hallway",
            description="A long hallway with a locked door to the north.",
            exits={"south": "entrance", "north": "guard_room"},
            locked_exits={"north": "red key"},
        )
        self.rooms["guard_room"] = Room(
            name="guard_room",
            description="A large room with weapon racks. A guard blocks the treasure!",
            has_guard=True,
            exits={"south": "hallway", "east": "treasure_room"},
        )
        self.rooms["treasure_room"] = Room(
            name="treasure_room",
            description="The treasure room! Gold coins and jewels sparkle in the light.",
            items=[Item("dragon's treasure", ItemType.TREASURE,
                        "A massive hoard of gold and gems", {"value": 1000})],
            exits={"west": "guard_room"},
        )
        self.current_room = self.rooms["entrance"]

    def reset(self, seed: Optional[int] = None):
        # 对应官方 reset：即使确定性地图不使用 seed，也会消耗一次随机数。
        if seed is None:
            seed = random.randint(0, 10000)
        self.__init__(seed=seed)
        return self.get_state_description()

    def get_state_description(self):
        room = self.current_room
        lines = [f"Room: {room.name}", room.description]
        if room.has_guard and not room.guard_defeated:
            lines.append("A guard blocks your way!")
        if room.items:
            lines.append("Items: " + ", ".join(x.name for x in room.items))
        lines.append("Exits: " + ", ".join(room.exits))
        lines.append("Inventory: " + ", ".join(x.name for x in self.inventory))
        lines.append(f"Moves: {self.moves}/{self.max_moves}")
        return "\n".join(lines)

    def get_available_actions(self) -> List[str]:
        actions = [f"go {direction}" for direction in self.current_room.exits]
        actions.extend(f"take {item.name}" for item in self.current_room.items)
        for item in self.inventory:
            actions.append(f"use {item.name}")
            actions.append(f"drop {item.name}")
        if self.current_room.has_guard and not self.current_room.guard_defeated:
            for item in self.inventory:
                if item.item_type == ItemType.WEAPON:
                    actions.append(f"attack with {item.name}")
        actions.extend(["look around", "check inventory"])
        if len(self.inventory) >= 2:
            actions.append("try crafting")
        return actions

    def _move(self, direction):
        room = self.current_room
        if direction not in room.exits:
            return f"You can't go {direction} from here.", -1
        if direction in room.locked_exits:
            needed = room.locked_exits[direction]
            if not any(x.name == needed for x in self.inventory):
                return f"The {direction} exit is locked. You need a {needed}.", -0.5
            del room.locked_exits[direction]
            self.current_room = self.rooms[room.exits[direction]]
            return f"You unlock the door with the {needed} and move {direction}.", 5
        if room.has_guard and not room.guard_defeated:
            return "A guard blocks your way! You must defeat them first.", -1
        self.current_room = self.rooms[room.exits[direction]]
        return f"You move {direction} to the {self.current_room.name}.", 1

    def _take_item(self, item_name):
        for item in self.current_room.items:
            if item.name.lower() == item_name.lower():
                self.current_room.items.remove(item)
                self.inventory.append(item)
                if item.item_type == ItemType.TREASURE:
                    reward = 100
                elif item.item_type == ItemType.KEY:
                    reward = 5
                elif item.item_type == ItemType.WEAPON:
                    reward = 3
                else:
                    reward = 2
                return f"You take the {item.name}.", reward
        return f"There's no {item_name} here.", -0.5

    def _drop_item(self, item_name):
        for item in self.inventory:
            if item.name.lower() == item_name.lower():
                self.inventory.remove(item)
                self.current_room.items.append(item)
                return f"You drop the {item.name}."
        return f"You don't have a {item_name}."

    def _use_item(self, item_name):
        for item in self.inventory:
            if item.name.lower() == item_name.lower():
                if item.item_type == ItemType.POTION:
                    self.inventory.remove(item)
                    if "healing" in item.name:
                        return "You drink the healing potion and feel refreshed!", 5
                    if "strength" in item.name:
                        self.active_effects["strength"] = 10
                        return "You feel a surge of power! Your attacks will be stronger.", 5
                elif item.item_type == ItemType.KEY:
                    return f"The {item.name} will be used automatically when needed.", 0
                else:
                    return f"You can't use the {item.name} right now.", -0.5
        return f"You don't have a {item_name}.", -0.5

    def _attack(self, weapon_name):
        room = self.current_room
        if not room.has_guard or room.guard_defeated:
            return "There's nothing to attack here.", -1
        weapon = next((x for x in self.inventory if x.name.lower() == weapon_name.lower()), None)
        if weapon is None:
            return f"You don't have a {weapon_name}.", -1
        if weapon.item_type != ItemType.WEAPON:
            return f"The {weapon_name} is not a weapon!", -1
        guard_type = "strong guard"
        if weapon.name in self.weapon_effectiveness:
            if guard_type in self.weapon_effectiveness[weapon.name]:
                room.guard_defeated = True
                return f"You defeat the {guard_type} with your {weapon.name}!", 20
            return f"Your {weapon.name} is not effective against the {guard_type}!", -2
        return f"Your {weapon.name} doesn't seem to work.", -1

    def _try_crafting(self):
        if len(self.inventory) < 2:
            return "You need at least two items to craft.", -0.5
        names = set(item.name for item in self.inventory)
        for recipe, product in self.crafting_recipes.items():
            if recipe.issubset(names):
                for ingredient in recipe:
                    for item in self.inventory[:]:
                        if item.name == ingredient:
                            self.inventory.remove(item)
                            break
                if product == "silver sword":
                    crafted = Item("silver sword", ItemType.WEAPON, "A gleaming silver blade")
                else:
                    crafted = Item(product, ItemType.TOOL, "A crafted item")
                self.inventory.append(crafted)
                return f"You successfully craft a {product}!", 10
        return "These items don't combine into anything useful.", -0.5

    def _check_victory(self):
        return any(item.name == "dragon's treasure" for item in self.inventory)

    def execute_action(self, action: str):
        """执行一步，返回 (反馈文本, 奖励, 本局是否结束)。"""
        if self.game_over:
            return "Game is already over.", 0, True
        self.moves += 1
        action = action.lower().strip()
        out_of_moves = self.moves >= self.max_moves
        reward = -0.5  # 每走一步都有一点惩罚，鼓励更短路径。

        if action.startswith("go "):
            result, delta = self._move(action[3:])
            reward += delta
        elif action.startswith("take "):
            result, delta = self._take_item(action[5:])
            reward += delta
        elif action.startswith("use "):
            result, delta = self._use_item(action[4:])
            reward += delta
        elif action.startswith("drop "):
            result = self._drop_item(action[5:])
        elif action.startswith("attack with "):
            result, delta = self._attack(action[12:])
            reward += delta
        elif action == "look around":
            result = self.get_state_description()
        elif action == "check inventory":
            if self.inventory:
                result = "Inventory: " + ", ".join(
                    f"{x.name} ({x.item_type.value})" for x in self.inventory
                )
            else:
                result = "Your inventory is empty."
        elif action == "try crafting":
            result, delta = self._try_crafting()
            reward += delta
        else:
            result = f"Unknown action: {action}"
            reward -= 1

        if self._check_victory():
            self.victory = True
            self.game_over = True
            reward += 100
            result += "\n\nVICTORY! You've collected the dragon's treasure!"
        if not self.game_over and out_of_moves:
            self.game_over = True
            reward -= 10
            result += "\n\nYou've run out of moves! Game over."
        return result, reward, self.game_over


# ============================================================
# 2. 表格 Q-learning 算法：内嵌官方确定性训练逻辑
# ============================================================
class QLearningAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_victories = []
        self.victories = 0

    def state_hash(self, game: TreasureHuntGame) -> str:
        # 保持原版状态结构，以便和官方学习曲线对照。
        state_parts = [
            game.current_room.name,
            tuple(sorted(item.name for item in game.inventory)),
            tuple(sorted(item.name for item in game.current_room.items)),
            tuple(sorted(game.current_room.locked_exits.items())),
            game.current_room.has_guard and not game.current_room.guard_defeated,
        ]
        return str(state_parts)

    def choose_action(self, game: TreasureHuntGame, training=True) -> str:
        actions = game.get_available_actions()
        if not actions:
            return "look around"
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        state = self.state_hash(game)
        values = {action: self.q_table[state][action] for action in actions}
        if all(v == 0 for v in values.values()):
            return random.choice(actions)
        return max(values, key=values.get)

    def update(self, state, action, reward, next_state, next_actions, done):
        current = self.q_table[state][action]
        if done:
            target = reward
        elif next_actions:
            best_next = max(self.q_table[next_state][a] for a in next_actions)
            target = reward + self.discount_factor * best_next
        else:
            target = reward
        self.q_table[state][action] = current + self.learning_rate * (target - current)

    def train_episode(self, game: TreasureHuntGame):
        game.reset()
        total_reward = 0.0
        steps = 0
        while not game.game_over:
            state = self.state_hash(game)
            action = self.choose_action(game, training=True)
            _, reward, done = game.execute_action(action)
            next_state = self.state_hash(game)
            next_actions = game.get_available_actions() if not done else []
            self.update(state, action, reward, next_state, next_actions, done)
            total_reward += reward
            steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_victories.append(int(game.victory))
        self.victories += int(game.victory)
        return total_reward, steps, game.victory

    def save(self, path: Path):
        record = {
            "q_table": {state: dict(actions) for state, actions in self.q_table.items()},
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "episodes": len(self.episode_rewards),
            "victories": self.victories,
        }
        with path.open("wb") as f:
            pickle.dump(record, f)



# ============================================================
# 3. 读取配置：复用 config.yaml / .env，但不依赖其他 .py
# ============================================================
LLM_SYSTEM = (
    "你正在玩一款文字寻宝游戏，目标是取得宝藏并成功通关。"
    "游戏包含未知机制；你并未获得游戏源码、隐藏规则或预设答案。"
    "只可根据当前可见游戏状态、合法动作、过去的行动反馈推断策略。"
    "特别注意：需要发现并利用道具，必要时探索特殊动作。"
    "你每次必须从当前提供的合法动作列表中选一项，action 必须与列表中的英文字符串完全一致。"
    "只输出一个 JSON 对象：{\"action\":\"合法动作原文\",\"reason\":\"一句简短中文说明\"}。"
    "不要加入 Markdown、额外正文或未知动作。失败时反思已知反馈，避免无意义重复。"
)


def find_root() -> Path:
    p = Path(__file__).resolve().parent
    for parent in [p, *p.parents]:
        if (parent / "config.yaml").is_file():
            return parent
    # 没有配置文件时仍能独立运行 --dry-run；正式运行需要 API Key
    return p


def read_cfg(root: Path, args) -> dict:
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    raw = {}
    if (root / "config.yaml").is_file():
        if yaml is None:
            raise RuntimeError("检测到 config.yaml 但缺少 PyYAML：pip install pyyaml")
        raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8-sig")) or {}

    common = raw.get("model") or {}
    exp = raw.get("experiment_8_2") or {}
    rl_old = raw.get("experiment_8_1") or {}

    def choose(flag, key, default):
        return flag if flag is not None else exp.get(key, default)

    cfg = {
        "model": choose(args.model, "model", common.get("name", "qwen3.7-flash-2026-07-15")),
        "base_url": os.getenv("QWEN_API_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "temperature": float(choose(None, "temperature", common.get("temperature", 0.3))),
        "top_p": float(choose(None, "top_p", common.get("top_p", 0.9))),
        "max_output_tokens": int(choose(args.max_output_tokens, "max_output_tokens", 768)),
        "api_timeout": int(choose(args.api_timeout, "api_timeout", 120)),
        "history_steps": int(choose(args.history_steps, "history_steps", 50)),
        "rl_episodes": int(choose(args.rl_episodes, "rl_episodes", rl_old.get("episodes", 10000))),
        "rl_eval_episodes": int(choose(args.rl_eval_episodes, "rl_eval_episodes", rl_old.get("eval_episodes", 100))),
        "checkpoint_interval": int(choose(None, "checkpoint_interval", rl_old.get("checkpoint_interval", 1000))),
        "seed": int(choose(args.seed, "seed", rl_old.get("seed", 42))),
        "learning_rate": float(choose(None, "learning_rate", rl_old.get("learning_rate", 0.2))),
        "discount_factor": float(choose(None, "discount_factor", rl_old.get("discount_factor", 0.99))),
        "epsilon": float(choose(None, "epsilon", rl_old.get("epsilon", 1.0))),
        "epsilon_decay": float(choose(None, "epsilon_decay", rl_old.get("epsilon_decay", 0.9995))),
        "epsilon_min": float(choose(None, "epsilon_min", rl_old.get("epsilon_min", 0.1))),
        "max_llm_steps": int(choose(args.max_llm_steps, "max_llm_steps", 50)),
        "output_dir": choose(args.output_dir, "output_dir", "char1/learning-from-experience/results"),
        "quiet": args.quiet,
    }
    for key in ("rl_episodes", "rl_eval_episodes", "checkpoint_interval", "max_llm_steps", "max_output_tokens", "history_steps", "api_timeout"):
        if cfg[key] < 1:
            raise ValueError(f"{key} 必须 >= 1")
    if not (0 <= cfg["epsilon_min"] <= cfg["epsilon"] <= 1):
        raise ValueError("epsilon 必须满足 0 <= epsilon_min <= epsilon <= 1")
    if not (0 < cfg["epsilon_decay"] <= 1):
        raise ValueError("epsilon_decay 必须满足 0 < epsilon_decay <= 1")
    if not (0 < cfg["learning_rate"] <= 1 and 0 <= cfg["discount_factor"] <= 1):
        raise ValueError("learning_rate 或 discount_factor 越界")
    return cfg


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 4. Q-learning：沿用实验 8-1 的完整算法并重新跑一组基线
# ============================================================
def run_qlearning(cfg: dict, out: Path):
    random.seed(cfg["seed"])
    agent = QLearningAgent(
        learning_rate=cfg["learning_rate"],
        discount_factor=cfg["discount_factor"],
        epsilon=cfg["epsilon"],
        epsilon_decay=cfg["epsilon_decay"],
        epsilon_min=cfg["epsilon_min"],
    )
    game = TreasureHuntGame()
    checkpoints = []
    n = cfg["rl_episodes"]
    win = cfg["checkpoint_interval"]
    started = time.perf_counter()

    with tqdm(total=n, desc="[1/3] Q-learning 训练", unit="局", dynamic_ncols=True) as bar:
        for i in range(1, n + 1):
            agent.train_episode(game)
            if i % win == 0 or i == n:
                recent_win = agent.episode_victories[-win:]
                recent_reward = agent.episode_rewards[-win:]
                snap = {
                    "episode": i,
                    "victory_rate": sum(recent_win) / len(recent_win),
                    "avg_reward": mean(recent_reward),
                    "q_table_size": len(agent.q_table),
                    "epsilon": agent.epsilon,
                }
                checkpoints.append(snap)
                bar.set_postfix_str(f"近{len(recent_win)}局胜率={snap['victory_rate']:.1%}, Q表={snap['q_table_size']}")
            bar.update(1)

    train_seconds = time.perf_counter() - started
    eval_game = TreasureHuntGame()
    wins, rewards, lengths, first_trace = 0, [], [], []
    t0 = time.perf_counter()
    m = cfg["rl_eval_episodes"]
    with tqdm(total=m, desc="[2/3] Q-learning 评估", unit="局", dynamic_ncols=True) as bar:
        for i in range(m):
            eval_game.reset()
            total = 0.0
            steps = 0
            while not eval_game.game_over:
                room = eval_game.current_room.name
                action = agent.choose_action(eval_game, training=False)
                feedback, reward, done = eval_game.execute_action(action)
                steps += 1
                total += reward
                if i == 0:
                    first_trace.append({"step": steps, "room": room, "action": action, "feedback": feedback, "reward": reward, "victory": eval_game.victory})
            wins += int(eval_game.victory)
            rewards.append(total)
            lengths.append(steps)
            bar.update(1)
            if (i + 1) % max(1, m // 10) == 0 or i + 1 == m:
                bar.set_postfix_str(f"胜率={wins / (i + 1):.1%}")

    baseline = {
        "algorithm": "tabular Q-learning",
        "training": {
            "episodes": n,
            "victories": agent.victories,
            "victory_rate": agent.victories / n,
            "last_checkpoint": checkpoints[-1],
            "q_table_size": len(agent.q_table),
            "training_seconds": round(train_seconds, 3),
        },
        "evaluation": {
            "episodes": m,
            "victories": wins,
            "victory_rate": wins / m,
            "avg_steps": mean(lengths),
            "avg_reward": mean(rewards),
            "std_steps": pstdev(lengths),
            "evaluation_seconds": round(time.perf_counter() - t0, 3),
        },
        "learning_curve": checkpoints,
        "first_evaluation_trajectory": first_trace,
    }
    write_json(out / "qlearning_baseline.json", baseline)
    agent.save(out / "rl_agent.pkl")
    with (out / "learning_curve.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "victory_rate", "avg_reward", "q_table_size", "epsilon"])
        w.writeheader()
        w.writerows(checkpoints)
    return baseline


# ============================================================
# 5. Qwen：每一步根据当前状态和已知经验选择合法动作
# ============================================================
def build_user_message(game: TreasureHuntGame, history: list, history_steps: int) -> str:
    if history:
        recent = history[-history_steps:]
        events = [
            f"第{x['step']}步 | 房间:{x['room_before']} | 动作:{x['action']} | "
            f"反馈:{x['feedback'].replace(chr(10), ' ')} | 奖励:{x['reward']:+.1f}"
            for x in recent
        ]
        # 如果窗口被截断，显式告诉模型只包含最近的经验，不假装完整记忆。
        history_text = "\n".join(events)
        if len(recent) < len(history):
            history_text = f"（只保留最近 {len(recent)} 步，之前 {len(history) - len(recent)} 步未包含）\n" + history_text
    else:
        history_text = "暂无（这是首次行动）"
    actions = game.get_available_actions()
    return (
        "【已知行动经验】\n" + history_text + "\n\n"
        "【当前游戏状态】\n" + game.get_state_description() + "\n\n"
        "【本轮唯一允许选择的动作】\n" + "\n".join(f"- {a}" for a in actions) + "\n\n"
        "请选出下一步动作，只输出 {\"action\":\"...\",\"reason\":\"...\"}。"
    )


def request_action(client, cfg, prompt: str, available_actions: list):
    """严格验收模型给出的动作。不可解析时不自动挑选动作。"""
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_output_tokens"],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    latency = time.perf_counter() - started
    choice = resp.choices[0]
    content = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    usage_data = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None
    record = {
        "response_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", cfg["model"]),
        "elapsed_seconds": round(latency, 3),
        "finish_reason": getattr(choice, "finish_reason", None),
        "raw_content": content,
        "usage": usage_data,
        "prompt": prompt,
        "action": None,
        "reason": None,
        "parse_error": None,
        "fallback_used": False,
    }
    try:
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("模型返回 JSON 的顶层不是对象")
        action = obj.get("action")
        if not isinstance(action, str):
            raise ValueError("JSON action 不是字符串")
        # 只做大小写/首尾空格规范化，不把无效动作替换成另一个动作。
        normalized = {a.strip().casefold(): a for a in available_actions}
        canonical = normalized.get(action.strip().casefold())
        if canonical is None:
            raise ValueError(f"动作不在合法列表中: {action!r}")
        record["action"] = canonical
        record["action_normalized"] = canonical != action
        record["reason"] = str(obj.get("reason", ""))[:500]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_llm(client, cfg: dict, out: Path):
    game = TreasureHuntGame(seed=cfg["seed"])
    game.reset(seed=cfg["seed"])
    # 原游戏最多 50 步，因此本地限制不可超过它。
    limit = min(game.max_moves, cfg["max_llm_steps"])
    actions_history, raw_calls = [], []
    total_reward = 0.0
    stopped_reason = "step_limit"
    start = time.perf_counter()
    path_trace = out / "llm_trajectory.json"
    path_calls = out / "llm_api_records.json"

    with tqdm(total=limit, desc="[3/3] Qwen 首局寻宝", unit="步", dynamic_ncols=True) as bar:
        for step in range(1, limit + 1):
            room = game.current_room.name
            inventory = [x.name for x in game.inventory]
            allowed = game.get_available_actions()
            user_input = build_user_message(game, actions_history, cfg["history_steps"])
            if not cfg["quiet"]:
                tqdm.write(f"[Qwen] 第 {step} 步，位置 {room}，正在请求模型决策...")
            try:
                call = request_action(client, cfg, user_input, allowed)
            except Exception as exc:
                # 不静默 fallback：记录错误后停止，防止虚假「首次通关」。
                raw_calls.append({
                    "step": step, "response_id": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status_code": getattr(exc, "status_code", None),
                    "error_body": str(getattr(exc, "body", ""))[:2000],
                    "fallback_used": False,
                    "prompt": user_input,
                })
                stopped_reason = "api_error"
                write_json(path_calls, raw_calls)
                write_json(path_trace, actions_history)
                tqdm.write(f"[API失败] 第 {step} 步：{type(exc).__name__}: {exc}")
                break

            call["step"] = step
            raw_calls.append(call)
            write_json(path_calls, raw_calls)
            if call["parse_error"]:
                stopped_reason = "invalid_action"
                tqdm.write(f"[模型动作无效] 第 {step} 步：{call['parse_error']}")
                break

            feedback, reward, done = game.execute_action(call["action"])
            total_reward += reward
            item = {
                "step": step,
                "room_before": room,
                "inventory_before": inventory,
                "available_actions": allowed,
                "action": call["action"],
                "reason": call["reason"],
                "feedback": feedback,
                "reward": reward,
                "done": done,
                "victory": game.victory,
                "room_after": game.current_room.name,
            }
            actions_history.append(item)
            write_json(path_trace, actions_history)
            bar.update(1)
            bar.set_postfix_str(f"房间={game.current_room.name}, 成功={int(game.victory)}")
            if not cfg["quiet"]:
                tqdm.write(f"  动作: {call['action']} | 奖励: {reward:+.1f} | 原因: {call['reason'][:90]}")
                tqdm.write(f"  反馈: {feedback.replace(chr(10), ' ')[:190]}")
            if done:
                stopped_reason = "victory" if game.victory else "game_over"
                break

    def token_field(name):
        return sum(int((r.get("usage") or {}).get(name) or 0) for r in raw_calls)

    result = {
        "agent": "Qwen in-context learning; model weights not updated",
        "model": cfg["model"],
        "environment": "same deterministic five-room treasure hunt as RL",
        "attempts": 1,
        "victory": bool(game.victory),
        "steps": len(actions_history),
        "max_steps": limit,
        "stopped_reason": stopped_reason,
        "total_reward": round(total_reward, 3),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "api_calls": len(raw_calls),
        "api_errors": sum(1 for r in raw_calls if r.get("error")),
        "parser_failures": sum(1 for r in raw_calls if r.get("parse_error")),
        "fallback_actions": sum(1 for r in raw_calls if r.get("fallback_used")),
        "prompt_tokens": token_field("prompt_tokens"),
        "completion_tokens": token_field("completion_tokens"),
        "total_tokens": token_field("total_tokens"),
        "token_statistics_available": all(r.get("usage") is not None for r in raw_calls),
    }
    write_json(out / "llm_summary.json", result)
    return result


# ============================================================
# 6. 可视化及主函数
# ============================================================
def save_visualizations(baseline: dict, llm: dict, out: Path):
    curve = baseline["learning_curve"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([r["episode"] for r in curve], [100 * r["victory_rate"] for r in curve], marker="o")
    ax.set_xlabel("Q-learning training episodes")
    ax.set_ylabel("Recent win rate (%)")
    ax.set_title("Q-learning learning curve")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [baseline["evaluation"]["avg_steps"], llm["steps"]]
    bars = ax.bar([f"Q-learning ({baseline['evaluation']['episodes']} eval avg)", "Qwen (first attempt)"], vals)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .2,
                f"{bar.get_height():.1f}", ha="center", va="bottom")
    ax.set_ylim(0, max(vals + [1]) * 1.25)
    ax.set_ylabel("Steps (finished or stopped)")
    ax.set_title(f"Different evaluation scopes | Qwen victory: {llm['victory']}")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="实验 8-2：单文件 Q-learning vs Qwen 首次寻宝对比")
    parser.add_argument("--model", help="临时指定 Qwen 模型，默认读取 config.yaml 的 model.name")
    parser.add_argument("--rl-episodes", type=int, help="Q-learning 训练局数，默认 10000")
    parser.add_argument("--rl-eval-episodes", type=int, help="Q-learning 评估局数，默认 100")
    parser.add_argument("--max-llm-steps", type=int, help="Qwen 最多执行游戏动作数，默认 50")
    parser.add_argument("--max-output-tokens", type=int, help="Qwen 每步输出的最大 tokens，默认 768")
    parser.add_argument("--api-timeout", type=int, help="单步模型 API 超时秒数，默认 120")
    parser.add_argument("--history-steps", type=int, help="给 Qwen 展示最近多少步经验，默认 50")
    parser.add_argument("--seed", type=int, help="Q-learning 随机种子，默认 42")
    parser.add_argument("--output-dir", help="结果根目录")
    parser.add_argument("--quiet", action="store_true", help="不打印逐步反馈（进度条仍显示）")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不消耗 API")
    args = parser.parse_args()

    root = find_root()
    cfg = read_cfg(root, args)
    output_root = Path(cfg["output_dir"])
    if not output_root.is_absolute():
        output_root = root / output_root
    print("项目根目录：", root)
    print("游戏：脚本内置五房间寻宝；无需其他代码文件")
    print("文本模型：", cfg["model"])
    print("文本 API 地址：", cfg["base_url"])
    print("Q-learning：", cfg["rl_episodes"], "局训练 +", cfg["rl_eval_episodes"], "局评估")
    print("Qwen：首次一局，最多", min(cfg["max_llm_steps"], 50), "步")
    print("输出根目录：", output_root)
    if args.dry_run:
        print("[配置检查] 通过，未发 API 请求，也未训练。")
        return
    if not cfg["api_key"]:
        raise EnvironmentError("找不到 DASHSCOPE_API_KEY；请在项目根目录 .env 中配置")
    if OpenAI is None:
        raise ImportError("请安装 OpenAI SDK：pip install openai")

    out = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment": "8-2-qlearning-versus-qwen-first-attempt",
        "model": cfg["model"],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {k: v for k, v in cfg.items() if k != "api_key"},
        "status": "running",
        "note": "Qwen replacement experiment, not an exact Kimi K3 reproduction; RL 100-episode evaluation and LLM first attempt have different sample sizes.",
    }
    write_json(out / "manifest.json", manifest)
    print("\n开始离线 Q-learning 基线训练...")
    baseline = run_qlearning(cfg, out)
    manifest["qlearning"] = baseline
    manifest["status"] = "rl_complete"
    write_json(out / "manifest.json", manifest)
    print(f"Q-learning 评估：{baseline['evaluation']['victory_rate']:.1%}，平均 {baseline['evaluation']['avg_steps']:.1f} 步")

    client = OpenAI(
        api_key=cfg["api_key"], base_url=cfg["base_url"],
        timeout=cfg["api_timeout"], max_retries=0,
    )
    print("\n开始 Qwen 首次寻宝：每步一轮模型调用，不更新模型参数。")
    llm = run_llm(client, cfg, out)
    comparison = {
        "qlearning": {
            "training_episodes": baseline["training"]["episodes"],
            "evaluation_episodes": baseline["evaluation"]["episodes"],
            "eval_win_rate": baseline["evaluation"]["victory_rate"],
            "eval_avg_steps": baseline["evaluation"]["avg_steps"],
            "q_table_states": baseline["training"]["q_table_size"],
            "training_seconds": baseline["training"]["training_seconds"],
        },
        "qwen": llm,
        "caution": "Do not equate a single Qwen victory/failure with a population win rate; Q-learning is a trained agent and Qwen uses pretrained world knowledge.",
    }
    write_json(out / "comparison.json", comparison)
    save_visualizations(baseline, llm, out)
    manifest["llm"] = llm
    manifest["status"] = "completed" if llm["api_errors"] == 0 and llm["parser_failures"] == 0 else "incomplete"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(out / "manifest.json", manifest)
    print("\n[实验完成]")
    print(f"Q-learning: 训练{baseline['training']['episodes']}局, 评估成功率{baseline['evaluation']['victory_rate']:.1%}, 平均{baseline['evaluation']['avg_steps']:.1f}步")
    print(f"Qwen: 首次挑战 {'成功' if llm['victory'] else '未成功'}, 已走{llm['steps']}步, 状态={llm['stopped_reason']}")
    print(f"Qwen: 调用{llm['api_calls']}次, Token合计{llm['total_tokens']}, 兜底动作{llm['fallback_actions']}次")
    print("结果保存在：", out)


if __name__ == "__main__":
    main()
