"""Assemble a self-contained, explicitly restricted candidate after evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.output);candidate=run/'candidate'
    status=json.loads((candidate/'status.json').read_text(encoding='utf-8'))
    if status['status']!='training_and_test_complete':raise ValueError('Training and test not completed')
    comparison=json.loads((candidate/'comparison.json').read_text(encoding='utf-8'))
    checks=json.loads((run/'engineering_checks.json').read_text(encoding='utf-8'))
    if not checks['passed']:raise ValueError('Engineering checks did not pass')
    if out.exists():raise FileExistsError('Delivery output already exists')
    out.mkdir(parents=True)
    for folder in ('model','runtime','evaluation','provenance'):(out/folder).mkdir()
    narrow=run/'refinement_last_narrow_bn_probe'
    narrow_info=json.loads((narrow/'diagnostic.json').read_text(encoding='utf-8'))
    (out/'model_narrow_2048').mkdir()
    for name in ('weights.pth','model_manifest.json','diagnostic.json'):
        shutil.copy2(narrow/name,out/'model_narrow_2048'/name)
    narrow_mf=json.loads((out/'model_narrow_2048/model_manifest.json').read_text(encoding='utf-8'))
    narrow_mf['deployment']={'scope':'optional profile for the known 2048-wide low-light cameras',
        'release_status':'experimental_training_domain_only','production_accepted':False,
        'independent_test_available':False,'training_diagnostic_dice':narrow_info['metrics']['dice'],
        'review_required':True,'do_not_apply_to_wide_7520_scans':True}
    narrow_mf['deployment_warning']='Experimental narrow-camera profile: no independent evaluation. Manually review every output.'
    (out/'model_narrow_2048/model_manifest.json').write_text(json.dumps(narrow_mf,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copytree(narrow/'previews',out/'evaluation/narrow_training_diagnostic')
    for name in ('weights.pth','model_manifest.json','training_config.json','history.json','test_result.json','comparison.json','status.json','validation_calibration.json'):
        shutil.copy2(candidate/name,out/'model'/name)
    for folder in ('test_evaluation','legacy_v2_test'):
        shutil.copytree(candidate/folder,out/'evaluation'/folder)
    for name in ('samples.json','audit.json'):
        shutil.copy2(run/'dataset'/name,out/'provenance'/name)
    shutil.copy2(run/'engineering_checks.json',out/'evaluation/engineering_checks.json')
    shutil.copy2(run/'raw_rgb_check.json',out/'evaluation/raw_rgb_check.json')
    shutil.copy2(run/'refinement/history.json',out/'provenance/refinement_history.json')
    shutil.copy2(run/'refinement/training_config.json',out/'provenance/refinement_training_config.json')
    shutil.copytree(ROOT/'geocore_mask',out/'runtime/geocore_mask',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'api',out/'runtime/api',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'runtime/tools').mkdir()
    for name in ('predict_core_mask_stream.py','predict_core_mask_batch.py','train_retraining_v3.py','prepare_retraining_v3.py','calibrate_retraining_v3.py','calibrate_narrow_v3.py','check_rgb_sources_v3.py','probe_narrow_bn_v3.py','export_refinement_last_v3.py','verify_application_v3.py','package_retraining_v3.py'):
        shutil.copy2(ROOT/'tools'/name,out/'runtime/tools'/name)
    (out/'runtime/tests').mkdir()
    shutil.copy2(ROOT/'tests/test_v3_inference.py',out/'runtime/tests/test_v3_inference.py')
    versions={}
    for name in ('torch','numpy','Pillow','scipy','scikit-image','GDAL'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='provided by conda/runtime'
    (out/'runtime_environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    new=next(iter(comparison['new_model'].values()));old=next(iter(comparison['legacy_v2'].values()))
    per_image=json.loads((candidate/'test_evaluation/metrics.json').read_text(encoding='utf-8'))['images']
    image_metrics=[next(iter(r['metrics'].values())) for r in per_image]
    median_dice=statistics.median(r['dice'] for r in image_metrics)
    worst_dice=min(r['dice'] for r in image_metrics)
    mf=json.loads((out/'model/model_manifest.json').read_text(encoding='utf-8'))
    mf['deployment']={'engineering_checks_passed':True,'evaluated_test_group':'ZKZ4-5',
        'release_status':'candidate_not_accepted_for_unattended_production','production_accepted':False,
        'acceptance_basis':'Engineering judgment from large orange-tray false positives and missed brown core; no predeclared numerical acceptance threshold.',
            'independent_test_dice':new['dice'],'independent_test_iou':new['iou'],
        'test_image_median_dice':median_dice,'test_image_min_dice':worst_dice,
        'supported_input':'uint8 RGB PNG and GDAL-readable RGB raster; ENVI RGB order from HDR',
        'long_raster_output':'full resolution mask.tif (0/255) and probability.tif (float32)',
        'unseen_domain_review_required':True}
    mf['deployment_warning']='Candidate only: held-out Dice 0.6808 with substantial tray false positives and missed core. Manual review required; not accepted for unattended production.'
    (out/'model/model_manifest.json').write_text(json.dumps(mf,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'release_status.json').write_text(json.dumps({'release_status':mf['deployment']['release_status'],
        'production_accepted':False,'engineering_checks_passed':True,
        'independent_test_dice':new['dice'],'independent_test_iou':new['iou'],
        'default_application_model_replaced':False,'manual_review_required':True,
        'remaining_requirement':'Additional representative training examples and fresh untouched boreholes/cameras for acceptance.'},ensure_ascii=False,indent=2),encoding='utf-8')
    text=f'''# 岩心 U-Net V3 重训候选模型（未通过通用生产验收）

打包时间：{datetime.now().isoformat(timespec='seconds')}。

**状态：已完成训练和工程验证，可人工复核试用；不可作为无人复核的通用生产模型。**
独立钻孔测试Dice仅{new['dice']:.4f}，存在橙色岩心箱误检及棕色岩心漏检。工程通过不代表识别质量通过。未覆盖或替换现有默认模型。本次没有预先约定数值验收线，未通过结论基于实际大面积误检/漏检，不是事后设定的通过率门槛。

使用119份人工标注样本：77训练、22验证（ZK6711）、20独立测试（ZKZ4-5）。同钻孔/潜在同孔采集没有跨集合。
最佳权重来自第{mf['training']['best_epoch']}轮，验证集选择阈值{mf['inference']['threshold']}。测试集未用于模型或阈值选择。

完成两阶段训练：首阶段34轮，低照度强化微调9轮（EMA滑动平均）。主候选模型采用首阶段第22轮最佳权重；窄幅低照度实验配置采用第二阶段EMA与仅基于训练低照度图像的BN统计校准。强化微调未超过首阶段的独立验证集成绩，因此未替换主候选权重。

`model` 是具有独立钻孔测试结果的主模型。`model_narrow_2048` 是前4组2048宽低照度采集的可选模型，其训练集诊断Dice由通用模型的0.8272提高到{narrow_info['metrics']['dice']:.4f}，但没有独立窄幅测试集，使用前需看叠加图，不能把这个数值称为泛化准确率。两个目录都使用同一套预测命令，只需替换--model-package参数。请勿将窄幅模型用于7520宽扫描。

| 独立测试指标 | 本次V3 | 旧V2对照 |
| --- | ---: | ---: |
| Dice | {new['dice']:.6f} | {old['dice']:.6f} |
| IoU | {new['iou']:.6f} | {old['iou']:.6f} |
| Precision | {new['precision']:.6f} | {old['precision']:.6f} |
| Recall | {new['recall']:.6f} | {old['recall']:.6f} |
| 背景像素误报率 | {new['false_positive_rate']:.6f} | {old['false_positive_rate']:.6f} |

对照定义：旧V2原归一化、512原像素窗口、Sigmoid后Softmax；新V3用{mf['inference']['source_tile_size']}原像素上下文缩放512推理。二者统一使用稳定的条带拼接实现且不做形态学后处理，因此这是可重复的受控对照，不是旧UI全部后处理的回放。

测试集逐图Dice中位数：{median_dice:.6f}，最低：{worst_dice:.6f}（包括纯负图，空真值存在任何误检时其Dice为0）。总体像素指标不能代替每张图的质量检查。最终部署窗口为{mf['inference']['source_tile_size']}像素；窗口、阈值与批归一化统计的选择过程见model/validation_calibration.json，测试集未参与这些选择。

## 使用方法

已验证的Python：`D:\\Users\\anaconda\\envs\\geo_env2\\python.exe`，PyTorch 2.8.0 + CUDA 12.8，RTX 5060 8GB。
不要使用不支持RTX 5060的旧pytorch环境。依赖版本见runtime_environment.json。

PowerShell中处理一幅ENVI DAT或RGB TIFF（将输入和输出替换成实际路径，输出需为空/不存在）：

```powershell
& 'D:\\Users\\anaconda\\envs\\geo_env2\\python.exe' '{out / 'runtime/tools/predict_core_mask_stream.py'}' `
  --model-package '{out / 'model'}' `
  --input 'H:\\路径\\RGB-示例.dat' `
  --output 'H:\\路径\\V3掩膜输出' --device cuda
```

批量PNG图块：

```powershell
& 'D:\\Users\\anaconda\\envs\\geo_env2\\python.exe' '{out / 'runtime/tools/predict_core_mask_batch.py'}' `
  --model-package '{out / 'model'}' `
  --input-dir 'H:\\路径\\images' `
  --output-dir 'H:\\路径\\V3图块输出' --device cuda
```

现有应用仓库已接入新模型清单中的source_tile_size和output_activation；在接口请求的model_package参数指定本包model目录。PNG保持既有mask/overlay/contour输出；DAT/TIFF使用流式GeoTIFF输出，预览是缩略图，不提供全幅PNG或全幅矢量轮廓。调用方应读取output_files.mask_tif。直接加载weights.pth却仍沿用旧归一化/512源窗口/Softmax会产生错误结果。
API模块需其原有服务依赖；本次仅验证服务函数调用，未启动HTTP服务器或验收前端。测试用pydantic 2.11.7位于原训练目录api_test_deps，命令行推理不依赖该包。

## 输出与工程限制

- mask.tif为原图尺寸的0/255掩膜，probability.tif为0到1的float32前景概率。保持原图变换和坐标参考（如原图存在）。
- RGB通道来自ENVI头文件default bands；设备的{{2,1,0}}按零基BGR转RGB。无波段顺序时需显式指定--bands，参数为一基RGB波段。本批RGB-20230910的三组与RGB-20260702组需添加 `--bands 1,2,3`，其与标注PNG的像素一致性已验证。
- 模型输入必须为uint8 RGB；其他位深不能未经校准直接用于该模型。
- 原始SHP与PNG未修改。多边形在原图范围内栅格化，保留人工定义。
- 大图CPU融合缓存随图宽和窗口高变化，不随整幅行数增长。推理中断会在metadata.json标明incomplete，不能把未完成TIFF用于生产。
- 未使用全局填洞/去小斑块，避免误删破碎岩心、填平真实间隙。工程测试详情见evaluation/engineering_checks.json。
- 应用接口与同进程数组推理的概率差为0；相对于原独立评估保存的掩膜，混合精度计算存在27/9,437,184个阈值邻近像素差异（最大距阈值0.000178），并非逐位一致。完整短DAT的流式与数组推理概率差为0。
- 宽幅原始DAT只检查前8192行，这段为背景，不能作为岩心识别精度或全钻孔吞吐率验证。
- 当前独立量化测试只覆盖ZKZ4-5；低分辨率2023/2026采集目前在训练集中，没有独立跨孔成绩。sbm仅含两份负样本，不能据此声称具备其岩石前景泛化能力。
- 新钻孔/新相机建议先检查代表性叠加图。标注边界与细碎岩心存在人工语义差异；本报告不承诺所有场景均提升。

## 查看结果和复现

evaluation/test_evaluation为本次模型逐图结果，legacy_v2_test为旧模型对照。预览从左至右为原图、预测叠加、误差；红色为误检、黄色为漏检。逐图计数和分组指标见metrics.json。
model/history.json保存训练曲线数据，model/training_config.json保存配置，provenance/samples.json保存来源分组、坐标和输入哈希。
中间恢复点保存在原训练目录 `{candidate / 'resume.pth'}`，未复制到轻量交付包。
文件完整性见SHA256SUMS.json。runtime含模型定义、预处理和预测入口，可独立于旧项目使用。

## 下一轮改进需要什么

1. 补充不同颜色岩心箱、湿/干岩心、棕色岩心和低照度场景的完整标注图块；既要含岩心与箱体交界，也要含纯箱底/标签/履带负样本。未勾画像素仍代表背景，不是未知区域。
2. 按钻孔/相机整组留出新验收集，不与训练图块重叠。当前ZKZ4-5已用于一次最终测试并揭示错误；以后若据此针对性优化，它只能作为已知诊断集，不能再声称是未见过的独立验收。
3. 数据充足后进行按钻孔分组交叉验证、难负样本增强及多随机种子对照；只在开发/验证集上选择训练配置和阈值。全部119张直接参与再拟合可以研究，但不会自动产生独立泛化证据，本次没有这样做。
4. 新模型需同时通过逐场景质量复核和实际完整DAT测试，再考虑替换默认模型。不要仅凭验证集Dice为0.9159或窄幅训练诊断Dice为0.9097做生产验收。
'''
    (out/'README_使用与评估.md').write_text(text,encoding='utf-8')
    inventory={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'package':str(out),'files':len(inventory),'new_test':new,'legacy_test':old},ensure_ascii=False))


if __name__=='__main__':main()
