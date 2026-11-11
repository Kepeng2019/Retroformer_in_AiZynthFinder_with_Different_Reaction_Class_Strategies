""" Module containing CLI tool calculate route distances and do clustering """
from __future__ import annotations
import argparse
import warnings
import time
import math
import re
import ast
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

import route_distances.lstm.defaults as defaults
from route_distances.route_distances import route_distances_calculator
from route_distances.clustering import ClusteringHelper
from route_distances.utils.type_utils import RouteDistancesCalculator
import os

# 预编译正则表达式模式
REMOVE_EMPTY_PATTERN = re.compile(r"'children':\s*\[\]")
COMMA_BRACE_PATTERN = re.compile(r',\s*}')
COMMA_BRACKET_PATTERN = re.compile(r',\s*]')
BRACE_COMMA_PATTERN = re.compile(r'{\s*,')
BRACKET_COMMA_PATTERN = re.compile(r'\[\s*,')

def remove_empty_children_fast(s: str) -> str:
    """优化的字符串清理函数"""
    cleaned = REMOVE_EMPTY_PATTERN.sub('', s)
    cleaned = COMMA_BRACE_PATTERN.sub('}', cleaned)
    cleaned = COMMA_BRACKET_PATTERN.sub(']', cleaned)
    cleaned = BRACE_COMMA_PATTERN.sub('{', cleaned)
    cleaned = BRACKET_COMMA_PATTERN.sub('[', cleaned)
    return cleaned

def parse_paths_batch(trees_series: pd.Series) -> List:
    """批量解析路径字符串"""
    paths_list = []
    for trees in trees_series:
        cleaned_str = remove_empty_children_fast(trees)
        paths = ast.literal_eval(cleaned_str.replace('true', 'True').replace('false', 'False'))
        paths_list.append(paths)
    return paths_list

def _calc_distances_batch(batch_data: pd.DataFrame, calculator: RouteDistancesCalculator) -> pd.DataFrame:
    """批量计算距离矩阵"""
    results = []
    
    for idx, row in batch_data.iterrows():
        if len(row.trees) == 1:
            results.append({
                'distance_matrix': [[0.0]],
                'distances_time': 0
            })
            continue
        
        time0 = time.perf_counter_ns()
        
        # 使用缓存的解析结果
        if 'parsed_paths' in row.index:
            paths = row['parsed_paths']
        else:
            cleaned_str = remove_empty_children_fast(row['trees'])
            paths = ast.literal_eval(cleaned_str.replace('true', 'True').replace('false', 'False'))
        
        distances = calculator(paths)
        
        results.append({
            'distance_matrix': distances.tolist(),
            'distances_time': (time.perf_counter_ns() - time0) * 1e-9,
        })
    
    return pd.DataFrame(results, index=batch_data.index)

def _do_clustering_batch(batch_data: pd.DataFrame, min_density: int) -> pd.DataFrame:
    """批量执行聚类"""
    results = []
    
    for idx, row in batch_data.iterrows():
        distance_matrix = row['distance_matrix']
        n_paths = len(row['trees']) if 'trees' in row.index else len(distance_matrix)
        
        if distance_matrix == [[0.0]] or n_paths < 3:
            results.append({'cluster_labels': [], 'cluster_time': 0})
            continue
        
        N = len(distance_matrix)
        # 优化的聚类数计算
        if N <= 10:
            nclusters = N
        elif N <= 100:
            nclusters = 10
        else:
            nclusters = min(int(math.sqrt(N)), int(N / min_density), 50)
        
        time0 = time.perf_counter_ns()
        labels = ClusteringHelper.cluster(
            distance_matrix, nclusters, max_clusters=0
        ).tolist()
        cluster_time = (time.perf_counter_ns() - time0) * 1e-9
        
        results.append({'cluster_labels': labels, 'cluster_time': cluster_time})
    
    return pd.DataFrame(results, index=batch_data.index)

def process_single_hdf5_file(file_info: Tuple[str, str, dict]) -> Tuple[str, bool]:
    """
    处理单个HDF5文件并直接保存为CSV
    
    Args:
        file_info: (file_path, deep_number, config)
    
    Returns:
        (filename, success_status)
    """
    file_path, deep_number, config = file_info
    
    try:
        # 读取单个文件
        data = pd.read_hdf(file_path, "table")
        
        if data is None or len(data) == 0:
            print(f"Warning: {file_path} is empty")
            return (os.path.basename(file_path), False)
        
        # 解析路径
        data['parsed_paths'] = parse_paths_batch(data['trees'])
        
        # 初始化计算器
        if config['only_clustering']:
            calculator = None
        elif config['model'] == "ted":
            calculator = route_distances_calculator("ted", content="both")
        else:
            calculator = route_distances_calculator(
                "lstm",
                model_path=config['model'],
                fp_size=config['fp_size'],
                lstm_size=config['lstm_size'],
            )
        

        def effective_diversity(dist_matrix, tau):
            N = dist_matrix.shape[0]
            selected = []

            for i in range(N):
                keep = True
                for j in selected:
                    if dist_matrix[i, j] < tau:
                        keep = False
                        break
                if keep:
                    selected.append(i)

            return len(selected)
        
        # 计算距离
        if not config['only_clustering']:
            batch_size = 1
            distance_results = []
            
            for start_idx in range(0, len(data), batch_size):
                end_idx = min(start_idx + batch_size, len(data))
                batch = data.iloc[start_idx:end_idx]
                batch_results = _calc_distances_batch(batch, calculator)
                distance_results.append(batch_results)
            
            distance_df = pd.concat(distance_results)
            matrix = np.array(distance_df.distance_matrix.tolist()[0])
            triu = matrix[np.triu_indices(len(matrix), k=1)]
            # sample = np.random.choice(triu, size=min(10000, len(triu)), replace=False)
            # tau = np.percentile(all_distances, p)
            taus = [2.92, 3.84, 4.93]
            for tau in taus:
                diversity = effective_diversity(matrix, tau)
                
                
                output_dir = config['output_dir']
                os.makedirs(output_dir, exist_ok=True)
                with open(config['output_dir'] + '/TED_diversity.txt', 'a+') as f:
                    # for val in sample:
                    f.write(f"{diversity}\n")

       
       
        #     data = data.assign(
        #         distance_matrix=distance_df.distance_matrix,
        #         distances_time=distance_df.distances_time,
        #     )
        
        # # 聚类
        # batch_size = 100
        # cluster_results = []
        
        # for start_idx in range(0, len(data), batch_size):
        #     end_idx = min(start_idx + batch_size, len(data))
        #     batch = data.iloc[start_idx:end_idx]
        #     batch_results = _do_clustering_batch(batch, config['min_density'])
        #     cluster_results.append(batch_results)
        
        # cluster_df = pd.concat(cluster_results)
        # data = data.assign(
        #     cluster_labels=cluster_df.cluster_labels,
        #     cluster_time=cluster_df.cluster_time,
        # )
        
        # 只保留需要的列
        # required_columns = ['cluster_labels', 'cluster_time']
        # 如果还有其他需要保留的原始列（如ID等），可以在这里添加
        # 这里假设原始数据中可能有索引或其他标识列，保留第一列作为标识
        # original_columns = [col for col in data.columns if col not in required_columns + ['parsed_paths', 'trees']]
        
        # 构建要保存的DataFrame
        # output_columns = required_columns
        # output_data = data[output_columns].copy()
        
        # output_data = pd.DataFrame({
        #     'cluster_labels': cluster_df['cluster_labels'],
        #     'cluster_time': cluster_df['cluster_time']
        # })
        # 保存为CSV文件
        # 生成输出文件名：原HDF5文件名（去掉.hdf5后缀）+ _results.csv
        # base_name = os.path.splitext(os.path.basename(file_path))[0]
        # output_filename = f"{base_name}_results.csv"
        
        # # 确定输出目录
        # # if config['output_dir']:
        # output_dir = config['output_dir']
      
        
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, output_filename)
        
        # 保存CSV
        # output_data.to_csv(output_path, index=False)
        
        # print(f"Successfully processed {file_path} -> {output_path} (rows: {len(output_data)})")
        
        # 删除临时列
        # del output_data
        
        return (os.path.basename(file_path), True)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return (os.path.basename(file_path), False)

def process_files_in_parallel(file_list: List[str], config: dict, 
                              n_workers: int = None) -> List[Tuple[str, bool]]:
    """
    并行处理多个文件，每个文件独立保存
    
    Args:
        file_list: 文件路径列表
        config: 配置字典
        n_workers: 并行进程数
    
    Returns:
        处理结果列表 (filename, success)
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(file_list), 8)  # 限制最大进程数
    
    print(f"Processing {len(file_list)} files with {n_workers} workers...")
    
    # 准备参数
    args_list = [(f, config.get('deep_number', 'unknown'), config) for f in file_list]
    
    # 使用进程池
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_hdf5_file, args): args[0] 
                         for args in args_list}
        
        # 收集结果
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), 
                          desc="Processing files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in future: {e}")
    
    return results

def main() -> None:
    """Entry-point for CLI tool"""
    
    args = _get_args()
    # target_path = '/home/kara_chen/workspace/Retrosynthesis/modelsmatter-main_ver2/result_retroformer_2026_03_03/'
    target_path = '/home/chenkepeng/extra_data_2/Retrosynthesis/modelsmatter-main_ver2/result_retroformer_2026_03_03/'
    target_path = '/home/kara_chen/workspace/Retrosynthesis/modelsmatter-main_ver2/result_retroformer_2026_03_03/' 
    # 获取所有processed目录
    file_paths = [path for path in os.listdir(target_path) 
                  if os.path.isdir(os.path.join(target_path, path)) and path.endswith('processed')]
    # file_paths = ['50k_class_3_processed']  # 可以修改为处理所有文件夹
    file_paths.sort()
    deep_numbers = ['depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5']
    deep_numbers = ['depth_6']  # 可以修改为处理所有深度
    
    # 配置参数
    config = {
        'model': args.model,
        'fp_size': args.fp_size,
        'lstm_size': args.lstm_size,
        'min_density': args.min_density,
        'only_clustering': args.only_clustering,
        'output_dir': args.output_dir,  # 添加输出目录配置
        'deep_number': None,  # 会在循环中设置
    }
    
    for file_path in file_paths:
        print(f"\n{'='*60}")
        print(f"Processing directory: {file_path}")
        print(f"{'='*60}")
        
        for deep_number in deep_numbers:
            print(f"\n--- Processing depth: {deep_number} ---")
            
            # 获取该深度下的所有HDF5文件（1000个文件）
            hdf5_dir = os.path.join(target_path, file_path)
            hdf5_files = [os.path.join(hdf5_dir, f) 
                          for f in os.listdir(hdf5_dir) 
                          if f.endswith('.hdf5') and deep_number in f]
            hdf5_files.sort()  # 确保文件顺序一致
            # hdf5_files = hdf5_files[:20]  # 只处理前1000个文件，测试用    
            
            if not hdf5_files:
                print(f"No files found for {deep_number}, skipping...")
                continue
            
            print(f"Found {len(hdf5_files)} files to process")
            
            # 更新配置
            config['deep_number'] = deep_number
            # 设置输出目录为当前目录下的结果文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config['output_dir'] = os.path.join(current_dir, file_path, f"{deep_number}_results")
            
            # 并行处理所有文件（每个进程处理一个文件，直接保存为CSV）
            start_time = time.time()
            results = process_files_in_parallel(hdf5_files, config, n_workers=4)
            
            # 统计处理结果
            success_count = sum(1 for _, success in results if success)
            fail_count = len(results) - success_count
            
            end_time = time.time()
            print(f"\nFinished processing {deep_number}")
            print(f"  Time: {(end_time - start_time)/60:.2f} minutes")
            print(f"  Success: {success_count}/{len(hdf5_files)} files")
            print(f"  Failed: {fail_count}/{len(hdf5_files)} files")
            print(f"  Results saved to: {config['output_dir']}")
            
            # 清理内存
            import gc
            gc.collect()

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Tool to calculate pairwise distances for AiZynthFinder output"
    )
    parser.add_argument("--files", nargs="+", required=False, 
                       default=[])
    parser.add_argument("--fp_size", type=int, default=defaults.FP_SIZE)
    parser.add_argument("--lstm_size", type=int, default=defaults.LSTM_SIZE)
    parser.add_argument("--model", required=False, 
                       default="download/chembl_10k_route_distance_model.ckpt")
    parser.add_argument("--only_clustering", action="store_true", default=False)
    parser.add_argument("--nclusters", type=int, default=1)
    parser.add_argument("--min_density", type=int, default=5)
    parser.add_argument("--output", default="finder_output_dist_raw_test.hdf5")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory for CSV files (default: auto-create 'results' folder)")
    return parser.parse_args()

if __name__ == "__main__":
    # 设置启动方法为spawn，避免Linux下的fork问题
    mp.set_start_method('spawn', force=True)
    main()
