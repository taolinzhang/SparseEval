import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
import random
import sys
import concurrent.futures
from functools import partial

# --- 依赖 ---
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import kendalltau
# 假设 irt 和 utils 库存在, 或者相关功能已包含在此文件中
# from irt import *
# from utils import *

# --- 全局随机种子设置 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了保证结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- MLP 模型定义 ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# --- 数据加载函数 ---
def load_base_data(scenario):
    """加载并预处理指定场景的基础数据。"""
    print("正在加载和预处理基础数据...")
    pt_path = f'./preprocess_data/{scenario}.pt'
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"数据文件未找到: {pt_path}")
    
    pt_data = torch.load(pt_path, map_location=torch.device('cpu'))
    matrix = pt_data['matrix']
    Y = matrix.numpy()
    scenarios_position = {scenario: np.arange(Y.shape[1])}
    
    accuracy_matrix = Y[:, scenarios_position[scenario]]

    if np.min(accuracy_matrix) >= 0 and np.max(accuracy_matrix) <= 1:
        print("将 0/1 矩阵转换为 -1/1 矩阵...")
        accuracy_matrix = 2 * accuracy_matrix - 1
    
    item_matrix = accuracy_matrix.T
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    
    base_data = {
        'Y': Y, 
        'scenarios_position': scenarios_position,
        'normalized_vectors': item_matrix / norms,
    }
    print("基础数据准备完毕。")
    return base_data

# --- 代理模型训练函数 ---
def train_proxy_model(X_train, y_train, num_anchors, learning_rate, partial_epochs, device):
    """一个轻量级训练函数，用于在优化循环中快速训练代理MLP。"""
    model = nn.Sequential(
        nn.Linear(num_anchors, 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    pbar = tqdm(range(partial_epochs), desc="  - 训练代理模型", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        loss = criterion(model(X_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        
    return model, criterion

# --- [重构后] sparse_eval 锚点生成策略 ---
def generate_anchors_mlp_high(vectors, Y_train, num_anchors, gpu_id, num_refinement_steps=10, partial_epochs=2000, learning_rate=5e-4, **kwargs):
    """
    使用预先分割好的训练集 Y_train 来迭代优化锚点集。
    """
    print(f"正在使用 'sparse_eval' 策略生成 {num_anchors} 个锚点...")
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print(f"sparse_eval 策略将使用设备: {device}")
    
    scenario = sys.argv[1]
    init_strategy_list = {
        'mmlu': 'random', 
        'winogrande': 'random',
        }
    init_strategy = init_strategy_list.get(kwargs.get('init_strategy', scenario), 'kmeans')

    # --- 1. 初始化锚点 ---
    if init_strategy == 'kmeans':
        print("  - [sparse_eval] 使用 KMeans 初始化锚点...")
        kmeans = KMeans(n_clusters=num_anchors, random_state=seed, n_init='auto')
        kmeans.fit(vectors)
        closest_points_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        current_anchors = list(np.unique(closest_points_indices))
        while len(current_anchors) < num_anchors:
            new_point = random.choice(range(vectors.shape[0]))
            if new_point not in current_anchors: 
                current_anchors.append(new_point)
    else:
        print("  - [sparse_eval] 使用随机策略初始化锚点...")
        current_anchors = random.sample(range(vectors.shape[0]), num_anchors)

    # --- 2. 准备训练数据 (数据已由 main 函数传入) ---
    # 代理模型的目标 y_labels 就是训练集的平均准确率
    y_train_labels = Y_train.mean(dim=1)

    # --- 3. 迭代优化循环 ---
    for step in range(num_refinement_steps):
        print(f"  - [sparse_eval] 开始第 {step+1}/{num_refinement_steps} 轮优化...")
        
        # a. 准备当前锚点的数据并训练代理模型 (使用训练集)
        X_train_data = Y_train[:, current_anchors]
        proxy_model, criterion = train_proxy_model(X_train_data, y_train_labels, len(current_anchors), learning_rate, partial_epochs, device)
        
        # b. 评估现有锚点的重要性 (在训练集上进行)
        proxy_model.eval()
        X_train_data.requires_grad_(True)
        proxy_model.zero_grad()
        predictions = proxy_model(X_train_data).squeeze()
        loss = criterion(predictions, y_train_labels)
        loss.backward()
        
        importance_scores = torch.abs(X_train_data.grad).mean(dim=0)
        X_train_data.requires_grad_(False)
        
        weakest_idx_in_anchors = torch.argmin(importance_scores).item()
        anchor_to_remove = current_anchors[weakest_idx_in_anchors]

        # c. 寻找最佳“挑战者”列 (在训练集上进行)
        with torch.no_grad():
            all_predictions = proxy_model(X_train_data).squeeze()
            error_vector = all_predictions - y_train_labels
            
            Y_train_centered = Y_train - 0.5
            potential_scores = torch.abs(Y_train_centered.T @ error_vector)
            
            potential_scores[current_anchors] = -1.0
            challenger_anchor = torch.argmax(potential_scores).item()

        print(f"    - 交换锚点: {anchor_to_remove} (影响力: {importance_scores[weakest_idx_in_anchors]:.4f}) -> {challenger_anchor} (潜力: {potential_scores[challenger_anchor]:.4f})")

        # d. 执行替换
        if challenger_anchor not in current_anchors:
            current_anchors[weakest_idx_in_anchors] = challenger_anchor
    
    print("sparse_eval 策略执行完毕。")
    return current_anchors

# --- [重构后] MLP 评估函数 ---
def gradient_descent_mlp_evaluation(Y_train, Y_valid, Y_test, anchor_points, learning_rate, num_epochs, patience, gpu_id):
    """使用预先分割好的数据集和最终的锚点集来训练和评估MLP模型。"""
    print("\n开始使用最终的锚点集训练和评估MLP模型...")
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print(f"最终评估将使用设备: {device}")

    # --- 1. 准备输入和目标 (数据已提前分割好并传入) ---
    X_train, X_valid, X_test = Y_train[:, anchor_points], Y_valid[:, anchor_points], Y_test[:, anchor_points]
    y_train, y_valid, y_test = Y_train.mean(dim=1), Y_valid.mean(dim=1), Y_test.mean(dim=1)

    # --- 2. 定义MLP模型 ---
    model = nn.Sequential(
        nn.Linear(len(anchor_points), 1)
    ).to(device)
    scenario = sys.argv[1]
    vlm_dataset_list = [
        'ai2d', 'blink', 'ccbench', 'mmmu', 'mmtbench', 'realworldqa', 
    ]
    llm_dataset_list = [
        'arc', 'gsm8k', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande'
    ]
    if scenario in vlm_dataset_list:
        print("is_vlm = True")
        model = nn.Sequential(
            nn.Linear(len(anchor_points), 32), nn.GELU(), nn.Dropout(0.1),
            # nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            # nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 1), nn.Sigmoid()
        ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. 训练与早停 ---
    patience_counter = 0
    best_valid_loss = float('inf')
    best_test_error = float('inf')
    best_valid_tau = -2.0
    best_test_tau = -2.0

    for epoch in tqdm(range(num_epochs), desc="最终模型训练"):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train).squeeze()
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred_valid = model(X_valid).squeeze()
                valid_loss = criterion(pred_valid, y_valid)
                valid_tau, _ = kendalltau(y_valid.cpu().numpy(), pred_valid.cpu().numpy())
                
                if valid_loss <= best_valid_loss and valid_tau >= best_valid_tau:
                    best_valid_loss = valid_loss
                    best_valid_tau = valid_tau
                    
                    pred_test = model(X_test).squeeze()
                    best_test_error = torch.abs(pred_test - y_test).mean()
                    test_tau, _ = kendalltau(y_test.cpu().numpy(), pred_test.cpu().numpy())
                    best_test_tau = test_tau
                    
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n早停触发于 Epoch {epoch}。")
                    break
    
    if best_test_error != float('inf'):
        print(f"\n训练完成。最佳测试平均绝对误差 (MAE): {best_test_error.item():.4f}, 最佳测试 Kendall's τ: {best_test_tau:.4f}")
        return {'best_test_error': best_test_error.item(), 'best_test_tau': best_test_tau}
    else:
        print("\n训练未收敛或未找到更优模型。")
        return None

# --- [重构后] 主函数 ---
def main():
    """主函数，执行数据加载、一次性数据分割、锚点生成和模型评估的完整流程。"""
    if len(sys.argv) < 2:
        print("用法: python your_script_name.py <scenario_name>")
        return
        
    scenario = sys.argv[1]
    
    # --- 统一的超参数设置 ---
    num_gpus = torch.cuda.device_count()
    gpu_id = 0 if num_gpus > 0 else None
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print(f"检测到 {num_gpus} 个可用的GPU。" if num_gpus > 0 else "未检测到可用的GPU，将使用CPU。")

    learning_rate = 5e-4
    num_epochs = 20000
    patience = 30
    num_anchors = int(sys.argv[2])
    num_refinement_steps = 10

    root = f"result_data/main_result/{scenario}/linear/{num_anchors}"
    result_file = f"{root}/main_result.pt"
    os.makedirs(root, exist_ok=True)
    
    try:
        dump_results = torch.load(result_file) if os.path.exists(result_file) else []
        if not isinstance(dump_results, list): dump_results = []
    except Exception:
        dump_results = []
        
    while len(dump_results) < 10:
        try:
            # 1. 加载数据
            base_data = load_base_data(scenario)

            # --- 2. 划分数据集 (在外部只执行一次) ---
            print("正在划分数据集...")
            Y = base_data['Y']
            scenarios_position = base_data['scenarios_position']
            Y_scenario_full = Y[:, scenarios_position[scenario]]
            
            rng = np.random.default_rng(seed=seed)
            rand_idx = rng.permutation(Y_scenario_full.shape[0])
            
            Y_scenario_shuffled = torch.tensor(Y_scenario_full[rand_idx], device=device, dtype=torch.float32)

            vlm_dataset_list = [
                'ai2d', 'blink', 'ccbench', 'mmmu', 'mmtbench', 'realworldqa', 
            ]
            llm_dataset_list = [
                'arc', 'gsm8k', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande'
            ]
            if scenario in vlm_dataset_list:
                cnt = 50
            elif scenario in llm_dataset_list:
                cnt = 200
            else:
                exit("未知场景，请检查场景名称。")
            Y_test, Y_train_valid = Y_scenario_shuffled[:cnt], Y_scenario_shuffled[cnt:]
            # cnt =  min(50, Y_train_valid.shape[0] - 100)
            Y_valid, Y_train = Y_train_valid[:cnt], Y_train_valid[cnt:]
            # Y_valid, Y_train = Y_train_valid, Y_train_valid
            print(f"数据集划分完毕: Train={Y_train.shape[0]}, Valid={Y_valid.shape[0]}, Test={Y_test.shape[0]}")
            
            # 3. 使用 sparse_eval 策略生成锚点 (传入分割好的训练集)
            anchor_points = generate_anchors_mlp_high(
                vectors=base_data['normalized_vectors'],
                Y_train=Y_train,
                num_anchors=num_anchors,
                gpu_id=gpu_id,
                num_refinement_steps=num_refinement_steps
            )
            
            # 4. 使用生成的锚点训练和评估最终模型 (传入所有分割好的数据集)
            results = gradient_descent_mlp_evaluation(
                Y_train=Y_train,
                Y_valid=Y_valid,
                Y_test=Y_test,
                anchor_points=anchor_points,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                patience=patience,
                gpu_id=gpu_id
            )
            
            if results:
                print("\n--- 最终结果 ---")
                print(f"场景: {scenario}, 锚点数: {num_anchors}")
                print(f"最佳测试误差 (MAE): {results['best_test_error']:.6f}")
                print(f"最佳测试 Kendall's τ: {results['best_test_tau']:.6f}")
                print("-----------------")
                
                result_to_save = {'mae': results['best_test_error'], 'kendalls_tau': results['best_test_tau']}
                dump_results.append(result_to_save)
                
                torch.save(dump_results, result_file)
                print(f"结果已保存。当前共 {len(dump_results)}/10 条记录。")

        except Exception as e:
            print(f"\n执行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            break # 如果发生错误，跳出循环

if __name__ == "__main__":
    main()