# 项目迁移指南

## 目录重命名

请将项目目录从 `2` 重命名为 `nlp_text_classification_project`

## Windows用户
在命令提示符中运行：
```cmd
cd E:\NLP_lab\project
ren 2 nlp_text_classification_project
```

## Linux/Mac用户
```bash
cd /path/to/NLP_lab/project
mv 2 nlp_text_classification_project
```

## 注意事项

1. 所有路径引用已修复
   - Python代码中的相对路径已更新
   - 报告中的图片路径已更新为 `assets/images/`
   - 所有交叉引用已添加

2. 运行项目时请进入项目根目录后：
   ```bash
   cd nlp_text_classification_project
   ```

3. 训练模型：
   ```bash
   cd src
   python main.py --lr 0.001 --dropout 0.5 --channels 100
   ```

4. 查看报告：
   ```bash
   open docs/report.pdf  # Mac
   start docs/report.pdf  # Windows
   xdg-open docs/report.pdf  # Linux
   ```

## 项目结构

```
nlp_text_classification_project/
├── src/                    # 源代码
├── assets/                 # 资源文件
│   └── images/            # 所有图片
├── docs/                   # 文档
├── dataset/               # 数据集
├── save_model/            # 模型保存
├── batch_runs/            # 实验结果
└── logs/                  # 日志文件
```