import json
from collections import defaultdict
import numpy as np
import os
from datetime import datetime
import pandas as pd

from one_step_diversity.d_translate_400k_for_effective_factor import diversity_400k


class RetroTreeAnalyzer:
    def __init__(self, json_file, threshold=0.85):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.threshold = threshold

        self.mol_id_to_smiles = {}
        for mol_id, mol_info in self.data['molecules'].items():
            self.mol_id_to_smiles[int(mol_id)] = mol_info['smiles']

        # 当前层父节点集合，用 mol_id 作为主键
        self.current_layer_parents = defaultdict(set)

        self.all_paths = []
        self.depth_stats = defaultdict(lambda: {
            'nodes': set(),
            'unique_molecules': set(),
            # 主键由 mol_smiles 改为 mol_id，避免 SMILES 碰撞导致合并
            'parent_child_stats': defaultdict(lambda: {
                'children': set(),
                'total_children': 0
            }),
            'reaction_types': set(),
            'reaction_distribution': defaultdict(int),
            'leaf_nodes': set(),
            # 新增：有 children，但没有一个合法反应边抵达合法分子孙节点
            'anomalous_current_nodes': set(),
        })

        self.parent_child_relations = []
        self.tree_structure = defaultdict(list)
        self.node_depths = {}

        self._analyze_tree()
        self.json_file = json_file

    @staticmethod
    def _is_valid_molecule_node(node):
        """分子节点必须含 mol 字段，且 mol 不为 None"""
        return isinstance(node, dict) and 'mol' in node and node['mol'] is not None

    @staticmethod
    def _is_valid_reaction_node(node):
        """直接子节点必须是反应节点：存在 reaction 字段，且 reaction 是真值 dict"""
        if not isinstance(node, dict):
            return False
        if 'reaction' not in node:
            return False
        reaction = node.get('reaction')
        return isinstance(reaction, dict) and reaction

    def _extract_reaction_type(self, reaction_data):
        if not reaction_data:
            return 'unknown'
        metadata = reaction_data.get('metadata', {})
        return metadata.get('policy_name', 'unknown')

    def _traverse_tree(self, node, current_path, depth):
        if not self._is_valid_molecule_node(node):
            return

        mol_id = node['mol']
        mol_smiles = self.mol_id_to_smiles.get(mol_id, 'unknown')

        self.node_depths[mol_id] = depth
        current_path = current_path + [(mol_id, mol_smiles, depth)]

        self.depth_stats[depth]['nodes'].add(mol_id)
        self.depth_stats[depth]['unique_molecules'].add(mol_smiles)

        children = node.get('children')
        if not children:
            self.depth_stats[depth]['leaf_nodes'].add(mol_id)

            path_info = {
                'path_id': len(self.all_paths),
                'molecules': [(mid, smi, d) for mid, smi, d in current_path],
                'mol_ids': [mid for mid, _, _ in current_path],
                'smiles': [smi for _, smi, _ in current_path],
                'length': depth,
                'leaf_smiles': mol_smiles,
                'leaf_mol_id': mol_id
            }
            self.all_paths.append(path_info)
            return

        # 记录该父节点实际产生的合法子分子
        valid_grandchildren_for_this_parent = []

        for child_node in children:
            if not self._is_valid_reaction_node(child_node):
                continue

            reaction_info = child_node['reaction']
            reaction_type = self._extract_reaction_type(reaction_info)

            self.depth_stats[depth]['reaction_types'].add(reaction_type)
            self.depth_stats[depth]['reaction_distribution'][reaction_type] += 1

            for grandchild in child_node.get('children', []):
                if not self._is_valid_molecule_node(grandchild):
                    continue

                child_mol_id = grandchild['mol']
                child_smiles = self.mol_id_to_smiles.get(child_mol_id, 'unknown')

                self.parent_child_relations.append({
                    'parent_depth': depth,
                    'parent_mol_id': mol_id,        # 改为 mol_id
                    'parent_smiles': mol_smiles,
                    'child_mol_id': child_mol_id,    # 改为 mol_id
                    'child_smiles': child_smiles,
                    'reaction_type': reaction_type,
                    'reaction_data': reaction_info
                })

                self.tree_structure[mol_id].append((child_mol_id, reaction_info))

                # 主键统一为 mol_id，SMILES 仅作 payload
                self.depth_stats[depth]['parent_child_stats'][mol_id]['children'].add(child_mol_id)
                self.depth_stats[depth]['parent_child_stats'][mol_id]['total_children'] += 1

                valid_grandchildren_for_this_parent.append(child_mol_id)
                self._traverse_tree(grandchild, current_path, depth + 1)

        # 显式标记当前层节点身份
        if valid_grandchildren_for_this_parent:
            self.current_layer_parents[depth].add(mol_id)
        else:
            # 有 children，但没有一条合法反应边抵达合法分子孙节点
            self.depth_stats[depth]['anomalous_current_nodes'].add(mol_id)

    def _analyze_tree(self):
        root = self.data['tree']
        self._traverse_tree(root, [], 0)

        self.branching_stats = {}
        for depth in range(max(self.depth_stats.keys()) + 1):
            stats = self.depth_stats[depth]
            parent_stats = stats['parent_child_stats']

            b_eff_values = []

            for parent_mol_id, child_stats in parent_stats.items():
                if self.threshold:
                    product_smiles = self.mol_id_to_smiles.get(parent_mol_id, 'unknown')
                    # 子节点用 mol_id 取 SMILES，传给 diversity_400k
                    reactants = [
                        self.mol_id_to_smiles.get(cid, 'unknown')
                        for cid in child_stats['children']
                    ]
                    kept = diversity_400k(
                        product_smile=product_smiles,
                        reactant_smiles=reactants,
                        threshold=self.threshold
                    )
                    unique_children = len(set(kept))
                else:
                    unique_children = len(child_stats['children'])

                b_eff_values.append(unique_children)

            childs = float(np.sum(b_eff_values)) if b_eff_values else 0.0

            self.branching_stats[depth] = {
                'depth': depth,
                'n_nodes': len(stats['nodes']),
                'n_parents': len(self.current_layer_parents[depth]),  # 真正的父分子数
                'n_anomalous': len(stats['anomalous_current_nodes']),
                'childs': childs,
                'n_reaction_types': len(stats['reaction_types']),
                'n_leaf_nodes': len(stats['leaf_nodes']),
            }

    def get_paths_by_length(self):
        paths_by_length = defaultdict(list)
        for path in self.all_paths:
            paths_by_length[path['length']].append(path)
        return paths_by_length

    def get_experiment_data(self):
        """返回实验1的数据：每层的统计信息"""
        data = []
        for depth in sorted(self.branching_stats.keys()):
            s = self.branching_stats[depth]
            data.append({
                'depth': depth,
                'n_nodes': s['n_nodes'],
                'nodes_as_parents': s['n_parents'],
                'nodes_as_leaves': s['n_leaf_nodes'],
                'children': round(s['childs'], 4),
                'effective_factor': round(s['childs']/s['n_nodes'], 4),
                'n_anomalous_nodes': s['n_anomalous'],
            })
        return data

    def check_invariants(self):
        """检查恒等式是否成立：n_nodes == n_parents + n_leaf_nodes + n_anomalous"""
        report = {}
        for depth in sorted(self.depth_stats.keys()):
            stats = self.depth_stats[depth]
            nodes = stats['nodes']
            parents = self.current_layer_parents[depth]
            leaves = stats['leaf_nodes']
            anomalous = stats['anomalous_current_nodes']

            report[depth] = {
                'n_nodes': len(nodes),
                'n_parents': len(parents),
                'n_leaf_nodes': len(leaves),
                'n_anomalous': len(anomalous),
                'expected': len(parents) + len(leaves) + len(anomalous),
                'actual': len(nodes),
                'is_exact_partition': (
                    nodes == parents | leaves | anomalous
                    and parents.isdisjoint(leaves)
                    and parents.isdisjoint(anomalous)
                    and leaves.isdisjoint(anomalous)
                ),
            }
        return report


def find_json_files(base_dir, pattern="fulltree.json"):
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(pattern):
                json_files.append(os.path.join(root, file))
    return json_files


def extract_file_id(filepath):
    """从文件路径中提取标识符"""
    basename = os.path.basename(filepath)
    return basename[-18:-14]


def process_multiple_files(json_files, threshold=0.85, output_dir="analysis_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_experiment = []

    for i, json_file in enumerate(json_files, 1):
        print(f"\n处理进度: {i}/{len(json_files)}")
        print(f"文件: {json_file}")

        try:
            analyzer = RetroTreeAnalyzer(json_file, threshold=threshold)

            # 打印不变量检查
            invariants = analyzer.check_invariants()
            for depth, inv in sorted(invariants.items()):
                status = "OK" if inv['is_exact_partition'] else "MISMATCH"
                print(f"  深度 {depth}: n_nodes={inv['actual']}, "
                      f"n_parents={inv['n_parents']}, n_leaves={inv['n_leaf_nodes']}, "
                      f"n_anomalous={inv['n_anomalous']}, "
                      f"expected={inv['expected']} [{status}]")

            exp_data = analyzer.get_experiment_data()
            for row in exp_data:
                row['source_file'] = extract_file_id(json_file)
            all_experiment.extend(exp_data)

        except Exception as e:
            print(f"错误: 处理 {json_file} 时失败: {e}")
            continue

    if all_experiment:
        df_exp = pd.DataFrame(all_experiment)
        # 将 source_file 移到第一列
        cols = ['source_file'] + [c for c in df_exp.columns if c != 'source_file']
        df_exp = df_exp[cols]
        exp_file = os.path.join(output_dir, f"experiment_results.csv")
        df_exp.to_csv(exp_file, index=False)
        print(f"\n实验1结果已保存: {exp_file}")

    return all_experiment


if __name__ == "__main__":

    target = ['400k_class_0', '400k_class_1', '400k_class_3', '400k_class_5', '400k_class_10']
    target = ['400k_class_1', '400k_class_3', '400k_class_5', '400k_class_10']
    thresholds = [0.85, None]
    for threshold in thresholds:
        for tar in target:
            base_dir = f"result_retroformer_2026_03_03/{tar}"
            output_dir = base_dir + '_effectivate_factor_analysis_{}'.format(
                threshold if threshold else 'no_dediversity'
            )
            os.makedirs(output_dir, exist_ok=True)
            pattern = "fulltree.json"

            json_files = find_json_files(base_dir, pattern)
            json_files.sort()
            # json_files = json_files[:4]
            print(f"找到 {len(json_files)} 个JSON文件")

            if json_files:
                process_multiple_files(json_files, threshold=threshold, output_dir=output_dir)
            else:
                print("未找到任何JSON文件")
            # break