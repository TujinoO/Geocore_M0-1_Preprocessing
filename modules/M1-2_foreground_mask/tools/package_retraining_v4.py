"""Package verified V4 with explicit development-only evidence and rollback."""
import argparse,json,shutil,importlib.metadata
from pathlib import Path
from PIL import Image,ImageDraw
from prepare_retraining_v3 import digest
from train_retraining_v3 import dump

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-trial',required=True)
    a=p.parse_args();run=Path(a.run);out=Path(a.output)
    new=json.loads((run/'final_evaluation/summary.json').read_text(encoding='utf-8'))
    old=json.loads((run/'baseline_evaluation/summary.json').read_text(encoding='utf-8'))
    checks=json.loads((run/'engineering_checks.json').read_text(encoding='utf-8'))
    assert checks['passed']
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(run/'calibrated',out/'model')
    mf=json.loads((out/'model/model_manifest.json').read_text(encoding='utf-8'))
    mf['deployment']={'engineering_checks_passed':True,'independent_test_available':False,
        'release_status':'development_validated_manual_review_required','unattended_universal_production_accepted':False,
        'input':'uint8 RGB','label':'foreground core=255, background=0','normalization':'as specified in model manifest'}
    dump(out/'model/model_manifest.json',mf)
    for source,target in [('final_evaluation','evaluation/V4'),('baseline_evaluation','evaluation/V3'),('dataset/qa','annotation_qa')]:shutil.copytree(run/source,out/target)
    (out/'provenance').mkdir()
    for filename in ('samples.json','audit.json','visual_review.json'):shutil.copy2(run/'dataset'/filename,out/'provenance'/filename)
    shutil.copy2(run/'trial_selection.json',out/'provenance/trial_selection.json')
    shutil.copy2(run/'experiment_notes.md',out/'provenance/experiment_notes.md')
    for trial in ('frozen_bn','replay_ema'):
        source=run/trial
        if source.exists():
            target=out/'provenance'/trial;target.mkdir()
            for filename in ('training_config.json','history.json','status.json','starting_baseline.json','executed_training_source.py'):
                if (source/filename).exists():shutil.copy2(source/filename,target/filename)
    for name in ('blend_50','blend_75','blend_90'):
        if (run/name).exists():
            dest=out/'provenance'/name;dest.mkdir()
            for filename in ('model_manifest.json','evaluation.json','status.json'):shutil.copy2(run/name/filename,dest/filename)
    shutil.copy2(run/'engineering_checks.json',out/'evaluation/engineering_checks.json')
    shutil.copytree(ROOT/'geocore_mask',out/'runtime/geocore_mask',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'api',out/'runtime/api',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'runtime/tools').mkdir();(out/'runtime/tests').mkdir()
    for name in ('predict_core_mask_stream.py','predict_core_mask_batch.py','prepare_retraining_v3.py','prepare_retraining_v4.py','train_retraining_v3.py','train_retraining_v4.py','blend_retraining_v4.py','select_retraining_v4.py','evaluate_retraining_v4.py','verify_retraining_v4.py','package_retraining_v4.py'):
        shutil.copy2(ROOT/'tools'/name,out/'runtime/tools'/name)
    shutil.copy2(ROOT/'tests/test_v3_inference.py',out/'runtime/tests/test_v3_inference.py')
    versions={}
    for name in ('torch','numpy','Pillow','scipy','scikit-image','GDAL'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='conda/runtime'
    dump(out/'runtime_environment.json',versions)
    negative_ids={'195','196','200','202','203','204'}
    negative_report={}
    for version,folder in [('V3','baseline_evaluation'),('V4','final_evaluation')]:
        records=json.loads((run/folder/'new_hard_training/metrics.json').read_text(encoding='utf-8'))['images']
        entries=[{'id':r['id'],**next(iter(r['metrics'].values()))} for r in records if r['id'] in negative_ids]
        fp=sum(r['fp'] for r in entries);total=sum(r['fp']+r['tn'] for r in entries)
        negative_report[version]={'fp':fp,'background_pixels':total,'false_positive_rate':fp/total,'images':entries}
    dump(out/'evaluation/new_negative_training_diagnostic.json',negative_report)
    wide_records=json.loads((run/'final_evaluation/old_wide_training/metrics.json').read_text(encoding='utf-8'))['images']
    background074=next(iter(next(r for r in wide_records if r['id']=='074')['metrics'].values()))
    cases=out/'evaluation/selected_comparisons';cases.mkdir()
    for scope,key in [('adaptation_dev_ZKZ4-5','145'),('adaptation_dev_ZKZ4-5','165'),('validation_ZK6711','110'),('new_hard_training','195'),('old_wide_training','074')]:
        panel=Image.new('RGB',(1536,1150),'white');draw=ImageDraw.Draw(panel)
        for i,(version,folder) in enumerate([('V3','baseline_evaluation'),('V4','final_evaluation')]):
            draw.text((8,i*575+5),version+' | '+scope+' | '+key,fill='black')
            with Image.open(run/folder/scope/f'{key}.jpg') as im:panel.paste(im,(0,i*575+25))
        panel.save(cases/f'{key}_V3_V4.jpg',quality=93)
    labels={'validation_ZK6711':'ZK6711 验证集（未参与梯度训练，参与选模）',
            'adaptation_dev_ZKZ4-5':'ZKZ4-5 旧图开发集（同孔，非独立）',
            'new_hard_training':'新增困难样本（训练内诊断）',
            'old_low_training':'旧低分辨率样本（训练内诊断）',
            'old_wide_training':'旧宽幅样本（训练内诊断）'}
    table=['| 数据范围 | 张数 | V3 Dice | V4 Dice | V3背景误报率 | V4背景误报率 |','|---|---:|---:|---:|---:|---:|']
    for key,label in labels.items():
        n=new['subsets'][key];o=old['subsets'][key]['metrics'];m=n['metrics']
        table.append(f"| {label} | {n['count']} | {o['dice']:.4f} | {m['dice']:.4f} | {o['false_positive_rate']:.4f} | {m['false_positive_rate']:.4f} |")
    stages=[]
    for trial in ('frozen_bn','replay_ema'):
        path=run/trial/'status.json'
        if path.exists():
            status=json.loads(path.read_text(encoding='utf-8'));stages.append(f"- {trial}：完成{status['last_epoch']}轮，状态{status['status']}。")
    text=f'''# V4 岩心前景掩膜：困难样本优化版

## 状态与结果

新增36张标注已核查并用于优化。共155张：113张训练、22张ZK6711验证、20张ZKZ4-5同孔开发评估。
6张新纯负样本为195、196、200、202、203、204。用户确认完成标注的消息作为本次入库依据；旧pending表未改写。

最终训练配置：{a.selected_trial}。该权重来自本轮已运行候选中的开发集选择结果，不代表所有可能方法的全局最优。
源窗口{new['source_tile_size']}像素，模型输入512×512，阈值{new['threshold']}。具体搜索结果见model/selection.json。

{chr(10).join(stages)}

{chr(10).join(table)}

这里的背景误报率是所有真值背景像素中被预测成岩心的比例，不是专门精细标注的“橙色箱体误报率”。Dice/IoU为聚合像素指标，不是逐图平均准确率。

6张新纯背景图上的误报像素比例由{negative_report['V3']['false_positive_rate']:.4%}降至{negative_report['V4']['false_positive_rate']:.4%}，但这些图已经参与训练，只能作为学习效果诊断，不能作为新场景成绩。

本轮不是所有场景都单调提升：ZK6711验证Dice由{old['subsets']['validation_ZK6711']['metrics']['dice']:.4f}降至{new['subsets']['validation_ZK6711']['metrics']['dice']:.4f}，旧宽幅训练内Dice也略降。灰黑色、沾泥箱体仍可能被误识别，110号是明确退步案例。纯背景074号出现{background074['fp']}个误检像素（约{background074['false_positive_rate']:.4%}）；空真值只要出现误检，其Dice就会为0，不能仅看该图Dice判断误检面积。见evaluation/selected_comparisons，包含改善与退步实例，并非只展示好例子。

重要：ZKZ4-5在本轮已用于困难样本训练与针对性开发，旧20张虽然没有参与梯度训练，也不能再称独立测试。ZK6711从V3起就是选模验证集。本轮没有全新独立验收集，不能宣称跨所有新钻孔/相机的生产精度已通过验收。建议先人工复核试用，原默认模型未替换。

## 如何使用

使用已验证环境 `D:\\Users\\anaconda\\envs\\geo_env2\\python.exe`（RTX5060需PyTorch 2.8.0+CUDA12.8环境，旧pytorch环境不适用）。

PowerShell命令：

```powershell
& 'D:\\Users\\anaconda\\envs\\geo_env2\\python.exe' '{out/'runtime/tools/predict_core_mask_stream.py'}' `
  --model-package '{out/'model'}' `
  --input 'H:\\路径\\RGB-原始影像.dat' `
  --output 'H:\\路径\\V4掩膜输出' --device cuda
```

输出目录必须为空或不存在。RGB-ZKZ4-5等有default bands={{2,1,0}}的头文件自动采用3、2、1波段；旧20230910/20260702无此字段的数据需追加`--bands 1,2,3`。不接受未经校准的非uint8输入。

PNG批量推理：

```powershell
& 'D:\\Users\\anaconda\\envs\\geo_env2\\python.exe' '{out/'runtime/tools/predict_core_mask_batch.py'}' `
  --model-package '{out/'model'}' --input-dir 'H:\\路径\\images' `
  --output-dir 'H:\\路径\\V4图块输出' --device cuda
```

在原应用接口中，将`model_package`指定到本包model目录。不能仅替换weights.pth却继续使用旧归一化或Softmax。API服务函数已经检查，未启动HTTP服务器或验收前端；若独立启动API还需原服务依赖。

## 输出、质量检查与回退

- `mask.tif`：原分辨率0/255岩心掩膜；`probability.tif`：float32前景概率。模型是二分类，不输出岩心箱的独立类别。
- 大幅DAT采用窗口读取和条带融合，内存不随全幅行数线性增加。metadata.json必须status=complete；不要使用中断输出。
- 不默认做全局填洞/去小斑块，避免把真实箱底间隙填满或删除破碎岩心。GPU混合精度可能使极少量阈值邻近像素随后端数值变化。
- 先看每种场景的叠加图，重点检查箱沿、把手、棕色岩心、低照度、小块岩心和标签。如果新场景仍有系统性误检，应补充新钻孔代表样本，并另留真正独立的验收数据。
- `evaluation/V3`与`evaluation/V4`提供同一批标注下的逐图对照：左原图，中预测叠加，右误差（红误检、黄漏检）。训练内结果不可代替独立泛化证据。
- 原V3包仍在`H:\\Data_M1-2_foreground_mask_U-Net\\02_重训候选模型_20260926`，需要回退时把model_package重新指向该包，不用删除任何内容。
- 原始PNG/SHP全部保留，训练过程前后哈希检查见evaluation/engineering_checks.json。当前36张新增标注已经真实用于训练，非空模板。

## 完整原始影像验证

已执行完整短幅RGB与完整ZKZ4-5 DAT推理，输出在原实验目录：
`{run/'raw_short_full'}` 与 `{run/'raw_ZKZ4_full'}`。
这是完整输入读取、拼接、坐标和输出格式的工程验证，不是对整钻孔逐像素精度的独立验收。

训练与校准记录保留在`{run}`。模型包内含源码、配置、样本来源/标签哈希、训练曲线数据、评估图与SHA256SUMS.json。
'''
    (out/'README_使用与评估.md').write_text(text,encoding='utf-8')
    dump(out/'release_status.json',mf['deployment'])
    inventory={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()}
    dump(out/'SHA256SUMS.json',inventory)
    print(json.dumps({'package':str(out),'files':len(inventory),'subsets':new['subsets']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
