(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char1/learning-from-experience/llm_comparison.py
项目根目录： /mnt/e/ai-agent-book
游戏：脚本内置五房间寻宝；无需其他代码文件
文本模型： qwen3.7-flash-2026-07-15
文本 API 地址： https://dashscope.aliyuncs.com/compatible-mode/v1
Q-learning： 10000 局训练 + 100 局评估
Qwen：首次一局，最多 50 步
输出根目录： /mnt/e/ai-agent-book/char1/learning-from-experience/results

开始离线 Q-learning 基线训练...
[1/3] Q-learning 训练: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:09<00:00, 1006.05局/s, 近1000局胜率=98.1%, Q表=142]
[2/3] Q-learning 评估: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 3393.12局/s, 胜率=100.0%]
Q-learning 评估：100.0%，平均 12.0 步

开始 Qwen 首次寻宝：每步一轮模型调用，不更新模型参数。
[Qwen] 第 1 步，位置 entrance，正在请求模型决策...                                                                                                                                                                                                 
  动作: take rusty sword | 奖励: +2.5 | 原因: 初始房间有武器，拾取可增加战斗力或作为工具，且当前无其他紧急需求。                                                                                                                                   
  反馈: You take the rusty sword.                                                                                                                                                                                                                  
[Qwen] 第 2 步，位置 entrance，正在请求模型决策...                                                                                                                                                                                                 
  动作: go north | 奖励: +0.5 | 原因: 当前房间出口为北和东，且未探索任何区域。向北移动是推进游戏进程、寻找宝藏或新线索的标准策略，避免在起始房间原地打转。                                                                                         
  反馈: You move north to the hallway.                                                                                                                                                                                                             
[Qwen] 第 3 步，位置 hallway，正在请求模型决策...                                                                                                                                                                                                  
  动作: look around | 奖励: -0.5 | 原因: 当前房间描述仅提及北边有锁住的门，未提及其他物品或线索。在尝试使用武器前，先仔细观察环境以寻找可能的钥匙、机关或隐藏路径是更稳妥的策略。                                                                  
  反馈: Room: hallway A long hallway with a locked door to the north. Exits: south, north Inventory: rusty sword Moves: 3/50                                                                                                                       
[Qwen] 第 4 步，位置 hallway，正在请求模型决策...                                                                                                                                                                                                  
  动作: go north | 奖励: -1.0 | 原因: 当前房间描述指出北边有一扇锁着的门，但作为探索步骤，尝试向北移动以触发后续交互（如开锁或发现钥匙）是推进游戏的关键。                                                                                         
  反馈: The north exit is locked. You need a red key.                                                                                                                                                                                              
[Qwen] 第 5 步，位置 hallway，正在请求模型决策...                                                                                                                                                                                                  
  动作: go south | 奖励: +0.5 | 原因: 北门锁住且需要红钥匙，当前无钥匙，只能向南返回入口寻找线索或道具。                                                                                                                                           
  反馈: You move south to the entrance.                                                                                                                                                                                                            
[Qwen] 第 6 步，位置 entrance，正在请求模型决策...                                                                                                                                                                                                 
  动作: go east | 奖励: +0.5 | 原因: 北出口被锁且需要红色钥匙，当前库存无钥匙，必须探索其他方向（东）以寻找钥匙或通关线索。                                                                                                                        
  反馈: You move east to the storage.                                                                                                                                                                                                              
[Qwen] 第 7 步，位置 storage，正在请求模型决策...                                                                                                                                                                                                  
  动作: take red key | 奖励: +4.5 | 原因: 之前在北边走廊被锁住的门需要红色钥匙，现在在储物间找到了它，必须拾取以便后续开门。                                                                                                                       
  反馈: You take the red key.                                                                                                                                                                                                                      
[Qwen] 第 8 步，位置 storage，正在请求模型决策...                                                                                                                                                                                                  
  动作: take magic crystal | 奖励: +1.5 | 原因: 当前房间内有未拾取的道具magic crystal，拾取它可能获得额外奖励或用于后续解谜/合成。                                                                                                                 
  反馈: You take the magic crystal.                                                                                                                                                                                                                
[Qwen] 第 9 步，位置 storage，正在请求模型决策...                                                                                                                                                                                                  
  动作: go west | 奖励: +0.5 | 原因: 当前房间出口只有西面，且已持有开启北方锁门的红钥匙，需返回走廊以使用钥匙继续推进。                                                                                                                            
  反馈: You move west to the entrance.                                                                                                                                                                                                             
[Qwen] 第 10 步，位置 entrance，正在请求模型决策...                                                                                                                                                                                                
  动作: go north | 奖励: +0.5 | 原因: 已持有红色钥匙，前往走廊尝试打开北面的锁门以继续推进。                                                                                                                                                       
  反馈: You move north to the hallway.                                                                                                                                                                                                             
[Qwen] 第 11 步，位置 hallway，正在请求模型决策...                                                                                                                                                                                                 
  动作: use red key | 奖励: -0.5 | 原因: 之前尝试向北移动时提示需要红色钥匙，现在背包中有红色钥匙，使用它应该能打开北边的门继续前进。                                                                                                              
  反馈: The red key will be used automatically when needed.                                                                                                                                                                                        
[Qwen] 第 12 步，位置 hallway，正在请求模型决策...                                                                                                                                                                                                 
  动作: go north | 奖励: +4.5 | 原因: 已持有红色钥匙，尝试向北移动以通过之前锁住的门。                                                                                                                                                             
  反馈: You unlock the door with the red key and move north.                                                                                                                                                                                       
[Qwen] 第 13 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: attack with rusty sword | 奖励: -2.5 | 原因: 守卫阻挡了通往宝藏的路，且当前持有武器，必须击败守卫才能继续前进。                                                                                                                            
  反馈: Your rusty sword is not effective against the strong guard!                                                                                                                                                                                
[Qwen] 第 14 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: use magic crystal | 奖励: -1.0 | 原因: 之前的攻击无效，尝试使用魔法水晶对抗守卫以清除障碍。                                                                                                                                                
  反馈: You can't use the magic crystal right now.                                                                                                                                                                                                 
[Qwen] 第 15 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: look around | 奖励: -0.5 | 原因: 当前攻击和道具使用均无效，且守卫阻挡去路。通过观察房间细节（如武器架）寻找可能的线索、隐藏物品或破解方法，是打破僵局的关键步骤。                                                                          
  反馈: Room: guard_room A large room with weapon racks. A guard blocks the treasure! A guard blocks your way! Exits: south, east Inventory: rusty sword, red key, magic crystal Moves: 15/50                                                      
[Qwen] 第 16 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: try crafting | 奖励: +9.5 | 原因: 普通攻击无效且魔法水晶暂时无法使用，尝试制作可能获得能击败守卫的强力武器或道具。                                                                                                                         
  反馈: You successfully craft a silver sword!                                                                                                                                                                                                     
[Qwen] 第 17 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: attack with silver sword | 奖励: +19.5 | 原因: 之前的铁剑无效，但刚合成的银剑奖励极高且针对守卫，应尝试使用它击败守卫以获取宝藏。                                                                                                          
  反馈: You defeat the strong guard with your silver sword!                                                                                                                                                                                        
[Qwen] 第 18 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: use silver sword | 奖励: -1.0 | 原因: 守卫仍在阻挡宝藏，且之前的反馈显示银剑能有效击败守卫，使用它可以清除障碍并继续推进。                                                                                                                 
  反馈: You can't use the silver sword right now.                                                                                                                                                                                                  
[Qwen] 第 19 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: look around | 奖励: -0.5 | 原因: 守卫已被击败，但状态显示其仍在阻挡宝藏。查看周围以确认宝藏位置或触发交互机制。                                                                                                                            
  反馈: Room: guard_room A large room with weapon racks. A guard blocks the treasure! Exits: south, east Inventory: red key, silver sword Moves: 19/50                                                                                             
[Qwen] 第 20 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: look around | 奖励: -0.5 | 原因: 当前状态显示守卫仍在阻挡宝藏，且之前的攻击反馈存在矛盾或守卫已重生，需重新确认房间细节和守卫状态以决定下一步策略。                                                                                        
  反馈: Room: guard_room A large room with weapon racks. A guard blocks the treasure! Exits: south, east Inventory: red key, silver sword Moves: 20/50                                                                                             
[Qwen] 第 21 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: look around | 奖励: -0.5 | 原因: 当前守卫仍在阻挡宝藏，且之前的攻击和物品使用反馈表明需要特定条件或机制。通过再次查看房间描述，确认是否有新的线索、可互动的武器架或其他隐藏机制，以便找到击败守卫或获取宝藏的方法。                        
  反馈: Room: guard_room A large room with weapon racks. A guard blocks the treasure! Exits: south, east Inventory: red key, silver sword Moves: 21/50                                                                                             
[Qwen] 第 22 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: look around | 奖励: -0.5 | 原因: 当前状态显示守卫仍在阻挡宝藏，且之前的攻击和物品使用尝试未成功或无效。在有限的步数内，需要重新审视环境以寻找新的线索或机制（如守卫的弱点、隐藏路径或特定交互条件），避免盲目重复无效                      
  反馈: Room: guard_room A large room with weapon racks. A guard blocks the treasure! Exits: south, east Inventory: red key, silver sword Moves: 22/50                                                                                             
[Qwen] 第 23 步，位置 guard_room，正在请求模型决策...                                                                                                                                                                                              
  动作: go east | 奖励: +0.5 | 原因: 守卫已被击败，宝藏应在东侧出口方向，继续探索以获取胜利。                                                                                                                                                      
  反馈: You move east to the treasure_room.                                                                                                                                                                                                        
[Qwen] 第 24 步，位置 treasure_room，正在请求模型决策...                                                                                                                                                                                           
  动作: take dragon's treasure | 奖励: +199.5 | 原因: 已到达宝藏室，直接拾取最终目标物品以通关。                                                                                                                                                   
  反馈: You take the dragon's treasure.  VICTORY! You've collected the dragon's treasure!                                                                                                                                                          
[3/3] Qwen 首局寻宝:  48%|███████████████████████████████████████████████████████████████████████████▎                                                                                 | 24/50 [00:30<00:33,  1.28s/步, 房间=treasure_room, 成功=1]

[实验完成]
Q-learning: 训练10000局, 评估成功率100.0%, 平均12.0步
Qwen: 首次挑战 成功, 已走24步, 状态=victory
Qwen: 调用24次, Token合计18536, 兜底动作0次
结果保存在： /mnt/e/ai-agent-book/char1/learning-from-experience/results/20261011_161323_232317