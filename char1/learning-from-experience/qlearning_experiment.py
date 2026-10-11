import argparse
import csv
import json
import pickle
import random
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


# ============================================================
# 1. 寻宝游戏环境
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
# 3. 训练、评估、留证、可视化
# ============================================================
def locate_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir, *script_dir.parents]:
        if (parent / "config.yaml").is_file():
            return parent
    # 没有项目 YAML 也可以独立运行：结果输出到脚本所在目录。
    return script_dir


def read_settings(root: Path, args) -> dict:
    user_cfg = {}
    yaml_path = root / "config.yaml"
    if yaml_path.is_file():
        if yaml is None:
            raise RuntimeError("发现 config.yaml，但未安装 PyYAML。请运行 pip install pyyaml")
        raw_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
        user_cfg = raw_cfg.get("experiment_8_1") or {}

    def pick(value, name, default):
        return value if value is not None else user_cfg.get(name, default)

    cfg = {
        "episodes": int(pick(args.episodes, "episodes", 10000)),
        "eval_episodes": int(pick(args.eval_episodes, "eval_episodes", 100)),
        "checkpoint_interval": int(pick(args.checkpoint_interval, "checkpoint_interval", 1000)),
        "seed": int(pick(args.seed, "seed", 42)),
        "learning_rate": float(pick(args.learning_rate, "learning_rate", 0.2)),
        "discount_factor": float(pick(args.discount_factor, "discount_factor", 0.99)),
        "epsilon": float(user_cfg.get("epsilon", 1.0)),
        "epsilon_decay": float(pick(args.epsilon_decay, "epsilon_decay", 0.9995)),
        "epsilon_min": float(pick(args.epsilon_min, "epsilon_min", 0.1)),
        "output_dir": pick(args.output_dir, "output_dir", "char1/learning-from-experience/results"),
    }
    for key in ("episodes", "eval_episodes", "checkpoint_interval"):
        if cfg[key] < 1:
            raise ValueError(f"{key} 必须至少为 1")
    if not 0 < cfg["learning_rate"] <= 1:
        raise ValueError("learning_rate 必须属于 (0, 1]")
    if not 0 <= cfg["discount_factor"] <= 1:
        raise ValueError("discount_factor 必须属于 [0, 1]")
    if not 0 <= cfg["epsilon_min"] <= cfg["epsilon"] <= 1:
        raise ValueError("epsilon / epsilon_min 必须满足 0 <= epsilon_min <= epsilon <= 1")
    if not 0 < cfg["epsilon_decay"] <= 1:
        raise ValueError("epsilon_decay 必须属于 (0, 1]")
    return cfg


def train_with_tqdm(agent: QLearningAgent, cfg):
    game = TreasureHuntGame()
    curve = []
    n = cfg["episodes"]
    window = cfg["checkpoint_interval"]
    start = time.perf_counter()
    with tqdm(total=n, desc="Q-learning 训练", unit="局", dynamic_ncols=True) as pbar:
        for episode in range(1, n + 1):
            agent.train_episode(game)
            if episode % window == 0 or episode == n:
                recent_wins = agent.episode_victories[-window:]
                recent_rewards = agent.episode_rewards[-window:]
                recent_steps = agent.episode_lengths[-window:]
                row = {
                    "episode": episode,
                    "victory_rate": sum(recent_wins) / len(recent_wins),
                    "avg_reward": mean(recent_rewards),
                    "avg_steps": mean(recent_steps),
                    "q_table_size": len(agent.q_table),
                    "epsilon": agent.epsilon,
                }
                curve.append(row)
                pbar.set_postfix_str(
                    f"近{len(recent_wins)}局胜率={row['victory_rate']:.1%}, "
                    f"Q表={row['q_table_size']}, ε={agent.epsilon:.3f}"
                )
            pbar.update(1)
    return {
        "episodes": n,
        "victories": agent.victories,
        "victory_rate": agent.victories / n,
        "training_seconds": round(time.perf_counter() - start, 4),
        "q_table_size": len(agent.q_table),
        "final_epsilon": agent.epsilon,
        "learning_curve": curve,
        "episode_victories": agent.episode_victories,
        "episode_rewards": agent.episode_rewards,
        "episode_lengths": agent.episode_lengths,
    }


def evaluate_with_tqdm(agent: QLearningAgent, cfg):
    game = TreasureHuntGame()
    n = cfg["eval_episodes"]
    wins = 0
    rewards = []
    lengths = []
    first_trajectory = []
    start = time.perf_counter()
    with tqdm(total=n, desc="训练后贪婪评估", unit="局", dynamic_ncols=True) as pbar:
        for episode in range(1, n + 1):
            game.reset()
            total_reward = 0.0
            steps = 0
            while not game.game_over:
                room = game.current_room.name
                inventory = [item.name for item in game.inventory]
                action = agent.choose_action(game, training=False)
                feedback, reward, done = game.execute_action(action)
                total_reward += reward
                steps += 1
                if episode == 1:
                    first_trajectory.append({
                        "step": steps,
                        "room_before": room,
                        "inventory_before": inventory,
                        "action": action,
                        "feedback": feedback,
                        "reward": reward,
                        "done": done,
                        "victory": game.victory,
                    })
            wins += int(game.victory)
            rewards.append(total_reward)
            lengths.append(steps)
            pbar.update(1)
            if episode % max(1, n // 10) == 0 or episode == n:
                pbar.set_postfix_str(f"评估胜率={wins / episode:.1%}")
    return {
        "episodes": n,
        "victories": wins,
        "victory_rate": wins / n,
        "avg_reward": mean(rewards),
        "std_reward": pstdev(rewards),
        "avg_steps": mean(lengths),
        "std_steps": pstdev(lengths),
        "evaluation_seconds": round(time.perf_counter() - start, 4),
        "rewards": rewards,
        "steps": lengths,
    }, first_trajectory


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_curve(curve, out_dir: Path):
    fields = ["episode", "victory_rate", "avg_reward", "avg_steps", "q_table_size", "epsilon"]
    with (out_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve)

    x = [row["episode"] for row in curve]
    win = [row["victory_rate"] * 100 for row in curve]
    rewards = [row["avg_reward"] for row in curve]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, win, marker="o")
    axes[0].set_title("Q-learning Treasure Hunt")
    axes[0].set_ylabel("Victory rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, rewards, marker="o")
    axes[1].set_xlabel("Training episodes")
    axes[1].set_ylabel("Mean reward")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="实验 8-1：单文件 Q-learning 寻宝训练（离线）")
    parser.add_argument("--episodes", "--rl-episodes", type=int, default=None, help="训练局数，默认 10000")
    parser.add_argument("--eval-episodes", type=int, default=None, help="评估局数，默认 100")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="曲线采样间隔，默认 1000")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认 42")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discount-factor", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--epsilon-min", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不执行训练")
    args = parser.parse_args()

    root = locate_project_root()
    cfg = read_settings(root, args)
    output_root = Path(cfg["output_dir"])
    if not output_root.is_absolute():
        output_root = root / output_root

    print(f"项目根目录：{root}")
    print("游戏环境：脚本内部 TreasureHuntGame（官方确定性规则）")
    print("学习算法：脚本内部 QLearningAgent（表格 Q-learning）")
    print(f"训练局数：{cfg['episodes']} | 评估局数：{cfg['eval_episodes']}")
    print(f"随机种子：{cfg['seed']} | 曲线采样间隔：{cfg['checkpoint_interval']}")
    print(f"结果目录：{output_root}")
    print("无需 API Key，不依赖其他项目 Python 文件。")
    if args.dry_run:
        print("[配置检查] 通过，未执行训练。")
        return

    random.seed(cfg["seed"])
    agent = QLearningAgent(
        learning_rate=cfg["learning_rate"],
        discount_factor=cfg["discount_factor"],
        epsilon=cfg["epsilon"],
        epsilon_decay=cfg["epsilon_decay"],
        epsilon_min=cfg["epsilon_min"],
    )
    out_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=False)

    print("\n[1/2] 开始 Q-learning 训练")
    train = train_with_tqdm(agent, cfg)
    print("\n[2/2] 开始训练后独立评估（不更新 Q 值）")
    evaluation, first_trajectory = evaluate_with_tqdm(agent, cfg)

    evidence = {
        "experiment": "8-1-qlearning-standalone",
        "environment": "official deterministic five-room treasure hunt",
        "algorithm": "tabular Q-learning",
        "seed": cfg["seed"],
        "settings": cfg,
        "training": train,
        "evaluation": evaluation,
    }
    save_json(out_dir / "experiment_results.json", evidence)
    save_json(out_dir / "evaluation_first_episode.json", first_trajectory)
    save_curve(train["learning_curve"], out_dir)
    agent.save(out_dir / "rl_agent.pkl")

    print("\n[训练学习曲线]")
    print(f"{'局数':>8} {'最近胜率':>10} {'Q表大小':>10} {'epsilon':>10}")
    for row in train["learning_curve"]:
        print(f"{row['episode']:>8} {row['victory_rate']:>9.1%} "
              f"{row['q_table_size']:>10} {row['epsilon']:>10.3f}")
    print("\n[最终结果]")
    print(f"训练整体胜率：{train['victory_rate']:.1%}")
    print(f"训练耗时：{train['training_seconds']:.2f} 秒")
    print(f"评估胜率：{evaluation['victory_rate']:.1%} ({evaluation['victories']}/{evaluation['episodes']})")
    print(f"评估平均通关/结束步数：{evaluation['avg_steps']:.2f}")
    print(f"Q 表状态数：{train['q_table_size']}")
    print(f"文件保存位置：{out_dir}")


if __name__ == "__main__":
    main()
