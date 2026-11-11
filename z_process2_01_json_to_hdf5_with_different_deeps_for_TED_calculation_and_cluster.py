import json
import pandas as pd
from collections import defaultdict
import os

def convert_with_cumulative_depths(input_json_path, output_base_path=None,numb=0):
    """
    将JSON文件转换为按累积深度拆分的多个HDF5文件
    
    Args:
        input_json_path: 输入的JSON文件路径
        output_base_path: 输出文件的基础路径
    """
    print('\n\n\n\n'+f"正在读取文件: {input_json_path}")
    
    # if output_base_path is None:
    #     output_base_path = input_json_path.replace('.json', '')
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    tree = tree_data['tree']
    molecules = tree_data['molecules']
    
    # print(f"根节点SMILES: {molecules[str(tree['mol'])]['smiles']}")
    
    # 提取所有完整路径并计算深度（反应步数）
    def extract_complete_paths(node, current_path=None):
        """提取所有完整路径，返回(节点ID序列, 路径深度)"""
        if current_path is None:
            current_path = []
        
        node_id = node['mol']
        current = current_path + [node_id]
        
        # 如果没有子节点，说明到达叶子节点，返回完整路径
        if not node.get('children') or len(node['children']) == 0:
            # 路径深度 = 节点数 - 1（反应步数）
            path_depth = len(current) - 1
            return [(current, path_depth)]
        
        paths = []
        # 遍历所有反应子节点
        for child in node.get('children', []):
            if 'children' in child:
                for reactant in child['children']:
                    sub_paths = extract_complete_paths(reactant, current)
                    paths.extend(sub_paths)
        
        return paths
    
    # 提取所有路径及其深度
    all_paths_with_depth = extract_complete_paths(tree)
    # print(f"找到 {len(all_paths_with_depth)} 条完整路径")
    
    # 按路径深度分组
    depth_to_paths = defaultdict(list)
    for node_ids, path_depth in all_paths_with_depth:
        depth_to_paths[path_depth].append(node_ids)
    
    # 打印各深度路径数量
    print(f"\n各深度路径数量(非累积):")
    all_depths = sorted(depth_to_paths.keys())
    for depth in all_depths:
        print(f"  深度 {depth}: {len(depth_to_paths[depth])} 条路径")
    
    # 构建路径树的函数
    def build_path_tree(node_ids):
        """根据节点ID序列构建目标格式的树"""
        def build_node(depth):
            if depth >= len(node_ids):
                return None
            
            node_id = node_ids[depth]
            node_info = molecules[str(node_id)]
            
            mol_node = {
                'type': 'mol',
                'hide': False,
                'smiles': node_info['smiles'],
                'is_chemical': True,
                'in_stock': False,
                'children': []
            }
            
            if depth < len(node_ids) - 1:
                next_node_id = node_ids[depth + 1]
                next_node_info = molecules[str(next_node_id)]
                
                reaction_node = {
                    'type': 'reaction',
                    'hide': False,
                    'smiles': f"{node_info['smiles']}>>{next_node_info['smiles']}",
                    'is_reaction': True,
                    'metadata': {
                        'reaction': 'unknown',
                        'policy_probability': 0,
                        'policy_probability_rank': 0,
                        'policy_name': 'extracted'
                    },
                    'children': [build_node(depth + 1)]
                }
                
                mol_node['children'] = [reaction_node]
            
            return mol_node
        
        return build_node(0)
    
    # 预构建所有路径树
    # print("\n预构建所有路径树...")
    path_trees_by_depth = {}
    for depth, node_ids_list in depth_to_paths.items():
        path_trees = []
        for node_ids in node_ids_list:
            path_tree = build_path_tree(node_ids)
            if path_tree:
                path_trees.append(path_tree)
        path_trees_by_depth[depth] = path_trees
        # print(f"  深度 {depth}: {len(path_trees)} 条路径树")
    
    # 生成累积深度的文件
    depth_stats = []
    cumulative_paths = []
    
    # 按深度从小到大处理，累积添加路径
    for target_depth in sorted(all_depths):
        # print(f"\n生成深度 <= {target_depth} 的累积文件...")
        
        # 添加当前深度的所有路径到累积列表
        cumulative_paths.extend(path_trees_by_depth[target_depth])
        
        # 将所有路径组合成一个列表
        all_paths_list = cumulative_paths.copy()
        
        # 转换为单引号字符串
        json_str = json.dumps(all_paths_list)
        single_quote_str = json_str.replace('"', "'")
        
        # 生成文件名
        depth_file_base = f"{output_base_path}_depth_{target_depth}"
        csv_file = f"{depth_file_base}.csv"
        hdf5_file = f"{depth_file_base}.hdf5"
        
        # 写入CSV
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('trees\n')
            f.write(f'"{single_quote_str}"\n')
        
        # 读取CSV并保存为HDF5
        df = pd.read_csv(csv_file)
        df.to_hdf(hdf5_file, key='table', mode='w')
        os.remove(csv_file)
        
        # print(f"  已保存: {hdf5_file} (包含 {len(all_paths_list)} 条路径)")
        
        # 记录统计信息
        depth_stats.append({
            'Target_Number': '{:04d}_{:04d}'.format(numb,numb+1),
            'max_depth': target_depth,
            'cumulative_paths': len(all_paths_list),
            'new_paths_at_this_depth': len(path_trees_by_depth[target_depth]),
            # 'No.': len(target_depth)*['{:04d}_{:04d}'.format(numb,numb+1)],
            # 'hdf5_file': hdf5_file
        })
    
    # 保存统计信息
    try:
        
        
        original_stats_file = 'result_retroformer_2026_03_03/' + output_base_path[:-47].split('/')[-1] + '_cumulative_stats.csv'
        df = pd.read_csv(original_stats_file)
    except Exception as e:
        df = pd.DataFrame()

    # stats_file = f"{output_base_path}_cumulative_stats.csv"
    stats_df = pd.concat([df, pd.DataFrame(depth_stats)], ignore_index=True)
    stats_df.to_csv(original_stats_file, index=False)
    
    # print(f"\n统计信息已保存到: {stats_file}")
    # print("\n累积深度文件统计:")
    # print(stats_df.to_string(index=False))
    
    return depth_stats

# 分析函数：查看所有路径的深度分布
def analyze_depth_distribution(input_json_path):
    """
    分析JSON文件中所有路径的深度分布
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    tree = tree_data['tree']
    molecules = tree_data['molecules']
    
    def collect_paths(node, current_path=None):
        if current_path is None:
            current_path = []
        
        node_id = node['mol']
        current = current_path + [node_id]
        
        if not node.get('children') or len(node['children']) == 0:
            return [(current, len(current) - 1)]
        
        paths = []
        for child in node.get('children', []):
            if 'children' in child:
                for reactant in child['children']:
                    sub_paths = collect_paths(reactant, current)
                    paths.extend(sub_paths)
        return paths
    
    all_paths = collect_paths(tree)
    
    # 统计深度分布
    from collections import Counter
    depth_counts = Counter([depth for _, depth in all_paths])
    
    print("\n深度分布分析:")
    print("-" * 50)
    cumulative = 0
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        cumulative += count
        print(f"深度 {depth}: {count:4d} 条路径 | 累积到深度 {depth}: {cumulative:4d} 条路径")
    
    print("-" * 50)
    print(f"总计: {len(all_paths)} 条路径")
    
    return depth_counts

if __name__ == "__main__":
    
    for i in range(1000):
        
        current_path = os.getcwd() + "/result_retroformer_2026_03_03/"
        
        paths = [path for path in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, path)) and 'processed' not in path]
        paths =  ['50k_class_0', '50k_class_1', '400k_class_0','400k_class_1']
        paths.sort()
        for path in paths:
            
            # if path == '50k_class_10':
            #     continue
            input_file = "result_retroformer_2026_03_03/{}/retroformer_{}_1000_mols_200_iters_from_{:04d}_to_{:04d}_fulltree.json".format(path,path,i,i+1)
            output_path_part1 = "result_retroformer_2026_03_03/{}/".format(path+'_processed')
            os.makedirs(output_path_part1, exist_ok=True)
            output_path_part2 = "retroformer_{}_1000_mols_200_iters_from_{:04d}_to_{:04d}_fulltree".format(path,i,i+1)
            output_path = output_path_part1 + output_path_part2
            
            
            
            try:
            # 首先分析深度分布
                # print("=" * 60)
                # print("深度分布分析")
                # print("=" * 60)
                # depth_counts = analyze_depth_distribution(input_file)
                
                # 然后执行累积深度的转换
                # print("\n" + "=" * 60)
                # print("生成累积深度文件")
                # print("=" * 60)
                depth_stats = convert_with_cumulative_depths(
                    input_file, 
                    output_base_path=output_path,
                    numb = i
                )
                # break
                # 验证累积关系
                # print("\n" + "=" * 60)
                # print("验证累积关系")
                # print("=" * 60)
                # for i, stat in enumerate(depth_stats):
                #     if i == 0:
                #         print(f"深度 <= {stat['max_depth']}: {stat['cumulative_paths']} 条路径")
                #     else:
                #         expected = depth_stats[i-1]['cumulative_paths'] + stat['new_paths_at_this_depth']
                #         print(f"深度 <= {stat['max_depth']}: {stat['cumulative_paths']} 条路径 "
                #             f"(预期: {expected})")
            except Exception as e:
                print(f"处理文件 {input_file} 时发生错误: {e}")
        # if i ==3:
        #     break