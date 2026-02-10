import numpy as np
import torch
import os
import random
import sys
from functools import partial

# --- 依赖 ---
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import kendalltau

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

# --- [新] 基于 KMeans 簇大小的锚点和权重生成策略 ---
def generate_anchors_and_weights_kmeans(vectors, num_anchors):
    """
    使用 KMeans 生成锚点，并直接使用簇的大小作为权重。
    """
    print(f"正在使用 KMeans 策略生成 {num_anchors} 个锚点及其权重...")
    kmeans = KMeans(n_clusters=num_anchors, random_state=random.randint(1,10000), n_init='auto')
    print("  - 正在对向量进行 KMeans 聚类...")
    kmeans.fit(vectors)

    # 1. 找到离簇中心最近的点作为锚点
    anchor_points, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
    
    # 确保锚点是唯一的，尽管在实践中很少出现重复
    unique_anchor_points = np.unique(anchor_points)
    # if len(unique_anchor_points) < num_anchors:
    while len(unique_anchor_points) < num_anchors:
        print(f"  - 警告：KMeans 只找到了 {len(unique_anchor_points)} 个唯一的锚点，少于期望的 {num_anchors} 个。")
        kmeans = KMeans(n_clusters=num_anchors, random_state=random.randint(1,10000), n_init='auto')
        print("  - 正在对向量进行 KMeans 聚类...")
        kmeans.fit(vectors)

        # 1. 找到离簇中心最近的点作为锚点
        anchor_points, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        unique_anchor_points = np.unique(anchor_points)
    anchor_points = unique_anchor_points.tolist()

    # 2. 计算每个簇的大小 (cluster size) 作为权重
    # kmeans.labels_ 存储了每个输入向量所属的簇的索引
    # np.bincount 会统计每个索引（即每个簇）出现的次数
    cluster_sizes = np.bincount(kmeans.labels_)
    
    # 过滤掉可能因锚点去重而消失的簇的权重
    # 注意：kmeans.labels_ 中的簇索引与 cluster_centers_ 的索引是一致的
    # pairwise_distances_argmin_min 返回的 anchor_points 的顺序与 cluster_centers_ 的顺序也是一致的
    # 因此，我们可以直接使用 cluster_sizes 作为权重数组
    weights = cluster_sizes.astype(np.float32)

    print(f"  - KMeans 锚点和权重生成完毕。共 {len(anchor_points)} 个锚点。")
    return anchor_points, weights

# --- [新] 直接使用 KMeans 权重进行评估的函数 ---
def direct_kmeans_evaluation(Y_test, anchor_points, weights):
    """
    直接使用 KMeans 的簇大小作为权重来评估模型性能，不使用MLP。
    """
    print("\n开始使用 KMeans 权重进行直接评估...")

    # 1. 准备输入和目标
    # X_test 是测试集中只包含锚点列的数据
    X_test = Y_test[:, anchor_points]
    # y_test_true 是测试集中每一行的真实平均值
    y_test_true = Y_test.mean(dim=1)

    # 2. 计算加权平均预测
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=Y_test.device)
    total_weight = torch.sum(weights_tensor)

    # y_pred = (X_test * weights_tensor).sum(dim=1) / total_weight
    # 使用矩阵乘法提高效率: Y_test[i, anchor_points] * weights -> y_pred[i]
    y_test_pred = torch.matmul(X_test, weights_tensor) / total_weight
    
    # 3. 计算评估指标
    test_mae = torch.abs(y_test_pred - y_test_true).mean().item()
    test_tau, _ = kendalltau(y_test_true.cpu().numpy(), y_test_pred.cpu().numpy())

    print(f"评估完成。")
    return {'best_test_error': test_mae, 'best_test_tau': test_tau}

# --- [重构后] 主函数 ---
def main():
    """主函数，执行数据加载、一次性数据分割、锚点生成和直接评估的完整流程。"""
    if len(sys.argv) < 2:
        print("用法: python your_script_name.py <scenario_name>")
        return
        
    scenario = sys.argv[1]
    
    # --- 统一的超参数设置 ---
    num_gpus = torch.cuda.device_count()
    gpu_id = 0 if num_gpus > 0 else None
    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    print(f"检测到 {num_gpus} 个可用的GPU。" if num_gpus > 0 else "未检测到可用的GPU，将使用CPU。")

    num_anchors = int(sys.argv[2])

    root = f"result_data/main_result/{scenario}/anchor_points/{num_anchors}"
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

            # --- 2. 划分数据集 (仅划分出测试集用于评估) ---
            print("正在划分数据集...")
            Y = base_data['Y']
            scenarios_position = base_data['scenarios_position']
            Y_scenario_full = Y[:, scenarios_position[scenario]]
            
            rng = np.random.default_rng(seed=seed + len(dump_results)) # 每次运行使用不同种子以获得不同数据分割
            rand_idx = rng.permutation(Y_scenario_full.shape[0])
            
            Y_scenario_shuffled = torch.tensor(Y_scenario_full[rand_idx], device=device, dtype=torch.float32)

            # 仅划分出测试集
            cnt = 200 
            Y_test = Y_scenario_shuffled[:cnt]
            print(f"数据集划分完毕: Test={Y_test.shape[0]}")
            
            # 3. 使用 KMeans 生成锚点和权重
            # KMeans 在完整的、归一化后的向量空间上运行
            anchor_points, weights = generate_anchors_and_weights_kmeans(
                vectors=base_data['normalized_vectors'],
                num_anchors=num_anchors
            )
            
            # 4. 使用生成的锚点和权重直接进行评估
            results = direct_kmeans_evaluation(
                Y_test=Y_test,
                anchor_points=anchor_points,
                weights=weights
            )
            
            if results:
                print("\n--- 最终结果 ---")
                print(f"场景: {scenario}, 锚点数: {num_anchors}")
                print(f"测试误差 (MAE): {results['best_test_error']:.6f}")
                print(f"测试 Kendall's τ: {results['best_test_tau']:.6f}")
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