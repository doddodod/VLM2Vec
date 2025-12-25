#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Bad Case 分析器
用于逐图分析图像预测结果，识别和分析 bad case
"""

import json
import os
import tempfile

# 在导入gradio之前设置缓存目录（避免权限问题）
user_cache_dir = os.path.join(os.path.expanduser("~"), ".gradio_cache")
os.makedirs(user_cache_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = user_cache_dir

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class BadCaseAnalyzer:
    def __init__(self):
        self.result_data = None
        self.gt_data = None
        self.image_list = []
        self.current_image_id = None
        
    def load_result_file(self, result_file: str) -> Tuple[bool, str]:
        """加载预测结果文件"""
        try:
            if not result_file or not os.path.exists(result_file):
                return False, f"文件不存在: {result_file}"
            
            with open(result_file, 'r', encoding='utf-8') as f:
                self.result_data = json.load(f)
            
            # 获取图片列表
            if 'per_image_top100_candidates' in self.result_data:
                self.image_list = sorted(
                    [int(img_id) for img_id in self.result_data['per_image_top100_candidates'].keys()],
                    key=lambda x: x
                )
            elif 'per_image_results' in self.result_data:
                self.image_list = sorted(
                    [item['image_id'] for item in self.result_data['per_image_results']],
                    key=lambda x: x
                )
            else:
                return False, "结果文件中未找到图片列表（需要 per_image_top100_candidates 或 per_image_results 字段）"
            
            return True, f"成功加载结果文件，共 {len(self.image_list)} 张图片"
            
        except Exception as e:
            return False, f"加载结果文件失败: {str(e)}"
    
    def load_gt_file(self, gt_file: str) -> Tuple[bool, str]:
        """加载GT文件"""
        try:
            if not gt_file or not os.path.exists(gt_file):
                return False, f"文件不存在: {gt_file}"
            
            with open(gt_file, 'r', encoding='utf-8') as f:
                self.gt_data = json.load(f)
            
            # 构建 image_id -> gt_data 的映射
            if isinstance(self.gt_data, list):
                self.gt_dict = {item['image_id']: item for item in self.gt_data}
            else:
                self.gt_dict = {}
            
            return True, f"成功加载GT文件，共 {len(self.gt_dict)} 张图片"
            
        except Exception as e:
            return False, f"加载GT文件失败: {str(e)}"
    
    def get_image_info(self, image_id: int) -> Dict:
        """获取指定图片的信息"""
        if not self.result_data:
            return None
        
        info = {
            'image_id': image_id,
            'image_path': None,
            'candidates': [],
            'gt_relations': [],
            'objects': []
        }
        
        # 获取预测候选
        image_id_str = str(image_id)
        if 'per_image_top100_candidates' in self.result_data:
            candidates = self.result_data['per_image_top100_candidates'].get(image_id_str, [])
            info['candidates'] = candidates
            
            # 从第一个候选获取图片路径（如果有）
            if candidates and 'image_path' in candidates[0]:
                info['image_path'] = candidates[0]['image_path']
        
        # 获取GT信息
        if hasattr(self, 'gt_dict') and image_id in self.gt_dict:
            gt_item = self.gt_dict[image_id]
            info['image_path'] = gt_item.get('image_path', info['image_path'])
            info['objects'] = gt_item.get('objects', [])
            info['gt_relations'] = gt_item.get('relations', [])
        
        return info
    
    def format_candidate(self, cand: Dict, rank: int) -> str:
        """格式化候选结果"""
        subject = cand.get('subject', 'N/A')
        object_name = cand.get('object', 'N/A')
        predicted = cand.get('predicted_predicate', 'N/A')
        similarity = cand.get('similarity', 0.0)
        has_gt = cand.get('has_gt', False)
        is_correct = cand.get('is_correct', False)
        gt_predicate = cand.get('gt_predicate', '')
        
        # 状态标记
        status = ""
        if has_gt:
            if is_correct:
                status = "✅ 正确"
            else:
                status = f"❌ 错误 (GT: {gt_predicate})"
        else:
            status = "⚠️ 无GT"
        
        return f"**Rank {rank}** | {subject} --[{predicted}]--> {object_name} | 相似度: {similarity:.4f} | {status}"
    
    def format_gt_relation(self, rel: Dict, objects: List[Dict]) -> str:
        """格式化GT关系"""
        subject_id = rel.get('subject_id', -1)
        object_id = rel.get('object_id', -1)
        predicate = rel.get('predicate', 'N/A')
        
        # 获取物体名称
        subject_name = next((obj['class_name'] for obj in objects if obj['id'] == subject_id), f"ID_{subject_id}")
        object_name = next((obj['class_name'] for obj in objects if obj['id'] == object_id), f"ID_{object_id}")
        
        return f"✅ {subject_name} --[{predicate}]--> {object_name}"
    
    def draw_bbox_on_image(self, image_path: str, objects: List[Dict], 
                          highlight_ids: Optional[List[int]] = None) -> Optional[Image.Image]:
        """在图片上绘制bbox，返回PIL Image对象"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            # 打开图片
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except:
                    font = ImageFont.load_default()
            
            # 为每个物体分配颜色
            colors = [
                (255, 0, 0),    # 红色
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 蓝色
                (255, 255, 0),  # 黄色
                (255, 0, 255),  # 洋红
                (0, 255, 255),  # 青色
                (255, 165, 0),  # 橙色
                (128, 0, 128),  # 紫色
            ]
            
            # 绘制每个物体的bbox
            for obj in objects:
                obj_id = obj.get('id', -1)
                class_name = obj.get('class_name', 'Unknown')
                bbox = obj.get('bbox', [])
                
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # 选择颜色（高亮的用更亮的颜色）
                if highlight_ids and obj_id in highlight_ids:
                    color = (255, 0, 0)  # 红色高亮
                    width = 3
                else:
                    color = colors[obj_id % len(colors)]
                    width = 2
                
                # 绘制bbox矩形
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                
                # 绘制标签背景
                label_text = f"{obj_id}:{class_name}"
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # 标签背景
                label_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
                draw.rectangle(label_bg, fill=color)
                
                # 绘制标签文字
                draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(255, 255, 255), font=font)
            
            # 直接返回PIL Image对象，让Gradio处理
            return img
            
        except Exception as e:
            print(f"绘制bbox失败: {str(e)}")
            # 失败时返回原图
            try:
                return Image.open(image_path).convert('RGB')
            except:
                return None
    
    def analyze_image(self, image_id: int, top_k: int = 20) -> Tuple[str, str, str, Optional[Image.Image]]:
        """分析单张图片"""
        if not self.result_data:
            return "请先加载结果文件", "", "", None
        
        info = self.get_image_info(image_id)
        if not info:
            return f"未找到图片 {image_id} 的信息", "", "", None
        
        # 显示图片（带bbox）
        image_path = info['image_path']
        objects = info['objects']
        image_display = None
        
        if image_path and os.path.exists(image_path):
            try:
                # 获取需要高亮的物体ID（出现在预测结果中的）
                highlight_ids = set()
                for cand in info['candidates'][:top_k]:
                    highlight_ids.add(cand.get('subject_id'))
                    highlight_ids.add(cand.get('object_id'))
                highlight_ids = list(highlight_ids)
                
                # 绘制bbox，返回PIL Image对象
                image_display = self.draw_bbox_on_image(image_path, objects, highlight_ids)
            except Exception as e:
                print(f"处理图片失败: {str(e)}")
                try:
                    if image_path and os.path.exists(image_path):
                        image_display = Image.open(image_path).convert('RGB')
                except:
                    image_display = None
        
        # 构建预测结果文本
        candidates = info['candidates'][:top_k]
        pred_text = f"## 预测结果 (Top-{len(candidates)})\n\n"
        
        if not candidates:
            pred_text += "无预测结果\n"
        else:
            for i, cand in enumerate(candidates, 1):
                pred_text += self.format_candidate(cand, i) + "\n\n"
        
        # 构建GT结果文本
        gt_text = f"## 真实标签 (GT)\n\n"
        objects = info['objects']
        gt_relations = info['gt_relations']
        
        if not gt_relations:
            gt_text += "无GT数据\n"
        else:
            for rel in gt_relations:
                gt_text += self.format_gt_relation(rel, objects) + "\n\n"
        
        # 构建Bad Case分析
        badcase_text = self._analyze_bad_cases(info, top_k)
        
        return pred_text, gt_text, badcase_text, image_display
    
    def _analyze_bad_cases(self, info: Dict, top_k: int) -> str:
        """分析Bad Case"""
        text = "## Bad Case 分析\n\n"
        
        candidates = info['candidates'][:top_k]
        gt_relations = info['gt_relations']
        objects = info['objects']
        
        if not gt_relations:
            text += "⚠️ 无GT数据，无法进行Bad Case分析\n"
            return text
        
        # 构建GT关系集合 (subject_id, object_id, predicate)
        gt_pairs = set()
        for rel in gt_relations:
            gt_pairs.add((
                rel.get('subject_id'),
                rel.get('object_id'),
                rel.get('predicate')
            ))
        
        # 统计预测结果
        predicted_pairs = set()
        false_positives = []  # 预测了但不在GT中
        false_negatives = []  # GT中有但预测错误或未预测到
        
        for cand in candidates:
            subject_id = cand.get('subject_id')
            object_id = cand.get('object_id')
            predicted = cand.get('predicted_predicate')
            
            if subject_id is not None and object_id is not None and predicted:
                pair_key = (subject_id, object_id, predicted)
                predicted_pairs.add(pair_key)
                
                # 检查是否在GT中
                if pair_key not in gt_pairs:
                    # 检查是否有GT但预测错误
                    has_gt = cand.get('has_gt', False)
                    if has_gt:
                        gt_pred = cand.get('gt_predicate', '')
                        false_positives.append({
                            'subject_id': subject_id,
                            'object_id': object_id,
                            'predicted': predicted,
                            'gt': gt_pred,
                            'similarity': cand.get('similarity', 0.0)
                        })
                    else:
                        false_positives.append({
                            'subject_id': subject_id,
                            'object_id': object_id,
                            'predicted': predicted,
                            'gt': None,
                            'similarity': cand.get('similarity', 0.0)
                        })
        
        # 找出漏检的GT关系（False Negative）
        for rel in gt_relations:
            subject_id = rel.get('subject_id')
            object_id = rel.get('object_id')
            predicate = rel.get('predicate')
            
            # 检查是否在Top-K预测中
            found = False
            for cand in candidates:
                cand_subj_id = cand.get('subject_id')
                cand_obj_id = cand.get('object_id')
                cand_pred = cand.get('predicted_predicate')
                
                if (cand_subj_id == subject_id and 
                    cand_obj_id == object_id and 
                    cand_pred == predicate):
                    found = True
                    break
            
            if not found:
                false_negatives.append({
                    'subject_id': subject_id,
                    'object_id': object_id,
                    'predicate': predicate
                })
        
        # 输出分析结果
        text += f"### 统计信息\n"
        text += f"- GT关系总数: {len(gt_pairs)}\n"
        text += f"- Top-{top_k}预测数: {len(predicted_pairs)}\n"
        text += f"- 错误预测 (False Positive): {len(false_positives)}\n"
        text += f"- 漏检关系 (False Negative): {len(false_negatives)}\n\n"
        
        # False Positive
        if false_positives:
            text += f"### ❌ 错误预测 (False Positive, 共{len(false_positives)}个)\n\n"
            for i, fp in enumerate(false_positives[:10], 1):  # 只显示前10个
                subject_name = next((obj['class_name'] for obj in objects if obj['id'] == fp['subject_id']), f"ID_{fp['subject_id']}")
                object_name = next((obj['class_name'] for obj in objects if obj['id'] == fp['object_id']), f"ID_{fp['object_id']}")
                
                if fp['gt']:
                    text += f"{i}. {subject_name} --[{fp['predicted']}]--> {object_name} (GT: {fp['gt']}, 相似度: {fp['similarity']:.4f})\n"
                else:
                    text += f"{i}. {subject_name} --[{fp['predicted']}]--> {object_name} (无GT, 相似度: {fp['similarity']:.4f})\n"
            
            if len(false_positives) > 10:
                text += f"\n... 还有 {len(false_positives) - 10} 个错误预测\n"
            text += "\n"
        
        # False Negative
        if false_negatives:
            text += f"### ⚠️ 漏检关系 (False Negative, 共{len(false_negatives)}个)\n\n"
            for i, fn in enumerate(false_negatives, 1):
                subject_name = next((obj['class_name'] for obj in objects if obj['id'] == fn['subject_id']), f"ID_{fn['subject_id']}")
                object_name = next((obj['class_name'] for obj in objects if obj['id'] == fn['object_id']), f"ID_{fn['object_id']}")
                text += f"{i}. {subject_name} --[{fn['predicate']}]--> {object_name}\n"
            text += "\n"
        
        if not false_positives and not false_negatives:
            text += "✅ 未发现Bad Case！\n"
        
        return text


# 创建全局分析器实例
analyzer = BadCaseAnalyzer()


def load_files(result_file: str, gt_file: str, auto_analyze: bool = False) -> Tuple[str, gr.Dropdown, gr.Button, gr.Button, str, str, str, Optional[Image.Image], str]:
    """加载文件并更新图片列表"""
    result_msg = ""
    gt_msg = ""
    
    if result_file:
        success, msg = analyzer.load_result_file(result_file)
        result_msg = f"结果文件: {msg}\n"
    
    if gt_file:
        success, msg = analyzer.load_gt_file(gt_file)
        gt_msg = f"GT文件: {msg}\n"
    
    # 更新图片列表下拉框
    choices = [str(img_id) for img_id in analyzer.image_list] if analyzer.image_list else []
    current_value = choices[0] if choices else None
    
    # 更新按钮状态
    prev_enabled = len(choices) > 1
    next_enabled = len(choices) > 1
    
    # 如果自动分析且加载成功，分析第一张图片
    pred_text = ""
    gt_text = ""
    badcase_text = ""
    image_display = None
    image_status = ""
    
    if auto_analyze and current_value:
        try:
            pred_text, gt_text, badcase_text, image_display, image_status = update_analysis(current_value, 20)
        except:
            pass
    
    return (result_msg + gt_msg, 
            gr.Dropdown(choices=choices, value=current_value),
            gr.Button(interactive=prev_enabled),
            gr.Button(interactive=next_enabled),
            pred_text, gt_text, badcase_text, image_display, image_status)


def update_analysis(image_id_str: str, top_k: int) -> Tuple[str, str, str, Optional[Image.Image], str]:
    """更新分析结果"""
    if not image_id_str:
        return "请选择图片", "", "", None, ""
    
    try:
        image_id = int(image_id_str)
        pred_text, gt_text, badcase_text, image_display = analyzer.analyze_image(image_id, top_k)
        
        # 获取当前图片索引和总数
        if analyzer.image_list:
            current_idx = analyzer.image_list.index(image_id) if image_id in analyzer.image_list else 0
            total = len(analyzer.image_list)
            status_text = f"图片 {current_idx + 1} / {total} (ID: {image_id})"
        else:
            status_text = f"图片 ID: {image_id}"
        
        return pred_text, gt_text, badcase_text, image_display, status_text
    except Exception as e:
        return f"分析失败: {str(e)}", "", "", None, ""


def navigate_image(direction: str, current_image_id_str: str, top_k: int = 20) -> Tuple[str, str, str, str, Optional[Image.Image], str]:
    """导航到上一张或下一张图片"""
    if not current_image_id_str or not analyzer.image_list:
        return "", "", "", "", None, ""
    
    try:
        current_id = int(current_image_id_str)
        current_idx = analyzer.image_list.index(current_id) if current_id in analyzer.image_list else 0
        
        if direction == "prev":
            new_idx = max(0, current_idx - 1)
        else:  # next
            new_idx = min(len(analyzer.image_list) - 1, current_idx + 1)
        
        new_image_id = analyzer.image_list[new_idx]
        new_image_id_str = str(new_image_id)
        
        # 更新分析结果
        pred_text, gt_text, badcase_text, image_display, status_text = update_analysis(new_image_id_str, top_k)
        
        return new_image_id_str, pred_text, gt_text, badcase_text, image_display, status_text
    except Exception as e:
        return current_image_id_str, f"导航失败: {str(e)}", "", "", None, f"错误: {str(e)}"


# 创建 Gradio 界面
with gr.Blocks(title="Bad Case 分析器") as demo:
    gr.Markdown("# 🔍 Bad Case 分析器")
    gr.Markdown("用于逐图分析图像预测结果，识别和分析 bad case")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📁 文件加载")
            result_file_input = gr.Textbox(
                label="预测结果文件路径",
                placeholder="例如: /public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000_filter_original.json",
                value="/public/home/wangby2025/plusLab/outputs/test_2000_recall/four_card_train_74k.json"
            )
            gt_file_input = gr.Textbox(
                label="GT文件路径（可选）",
                placeholder="例如: /public/home/xiaojw2025/Data/embedding_similarity/vlm2vec_qwen2vl/result_recall_2000_filter_original.json",
                value="/public/home/wangby2025/plusLab/VLM2Vec/infer/test_2000_images.json"
            )
            load_btn = gr.Button("加载文件", variant="primary")
            load_status = gr.Textbox(label="加载状态", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("## 🖼️ 图片选择")
            with gr.Row():
                prev_btn = gr.Button("◀ 上一张", variant="secondary")
                next_btn = gr.Button("下一张 ▶", variant="secondary")
            image_dropdown = gr.Dropdown(
                label="选择图片ID",
                choices=[],
                value=None,
                interactive=True
            )
            image_status = gr.Textbox(label="图片信息", interactive=False, value="")
            top_k_slider = gr.Slider(
                label="显示Top-K预测结果",
                minimum=5,
                maximum=100,
                value=20,
                step=5
            )
            analyze_btn = gr.Button("分析图片", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_display = gr.Image(label="图片预览", type="pil")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("预测结果"):
                    pred_output = gr.Markdown(label="预测结果")
                
                with gr.Tab("真实标签"):
                    gt_output = gr.Markdown(label="真实标签")
                
                with gr.Tab("Bad Case分析"):
                    badcase_output = gr.Markdown(label="Bad Case分析")
    
    # 绑定事件
    load_btn.click(
        fn=lambda rf, gf: load_files(rf, gf, auto_analyze=True),
        inputs=[result_file_input, gt_file_input],
        outputs=[load_status, image_dropdown, prev_btn, next_btn, pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    # 自动加载默认文件
    demo.load(
        fn=lambda rf, gf: load_files(rf, gf, auto_analyze=True),
        inputs=[result_file_input, gt_file_input],
        outputs=[load_status, image_dropdown, prev_btn, next_btn, pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    analyze_btn.click(
        fn=update_analysis,
        inputs=[image_dropdown, top_k_slider],
        outputs=[pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    image_dropdown.change(
        fn=update_analysis,
        inputs=[image_dropdown, top_k_slider],
        outputs=[pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    top_k_slider.change(
        fn=update_analysis,
        inputs=[image_dropdown, top_k_slider],
        outputs=[pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    # 导航按钮
    prev_btn.click(
        fn=lambda img_id, k: navigate_image("prev", img_id, k),
        inputs=[image_dropdown, top_k_slider],
        outputs=[image_dropdown, pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    next_btn.click(
        fn=lambda img_id, k: navigate_image("next", img_id, k),
        inputs=[image_dropdown, top_k_slider],
        outputs=[image_dropdown, pred_output, gt_output, badcase_output, image_display, image_status]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 使用说明")
    gr.Markdown("""
    1. **加载文件**: 输入预测结果文件路径（必需）和GT文件路径（可选），点击"加载文件"
    2. **选择图片**: 从下拉列表中选择要分析的图片ID
    3. **设置Top-K**: 调整滑块设置要显示的Top-K预测结果数量
    4. **查看分析**: 
       - **预测结果**标签页：显示Top-K预测结果，标记正确/错误
       - **真实标签**标签页：显示GT关系
       - **Bad Case分析**标签页：分析错误预测(False Positive)和漏检关系(False Negative)
    """)


if __name__ == "__main__":
    # 可以通过环境变量指定端口
    port = int(os.environ.get("GRADIO_SERVER_PORT", 6660))
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )

