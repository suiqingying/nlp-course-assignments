# 期末报告模板

## 模板简介

这是一个专为中国科学院大学自然语言处理课程设计的期末报告LaTeX模板，采用专业的学术论文格式，包含完整的页眉页脚、标题样式和排版设计。

## 模板特点

- ✅ **官方格式**: 符合中国科学院大学试题专用纸格式要求
- ✅ **中文支持**: 使用ctex宏包，完美支持中文排版
- ✅ **专业排版**: 自定义标题样式、页眉页脚、颜色主题
- ✅ **数学公式**: 支持复杂数学公式和符号
- ✅ **图表支持**: 包含图片、表格、子图等环境
- ✅ **超链接**: 自动生成目录书签和交叉引用
- ✅ **开箱即用**: 包含完整示例内容，可直接修改使用

## 模板结构

```latex
\documentclass[11pt, a4paper]{article}

% 核心宏包
- ctex          % 中文支持
- geometry      % 页面设置
- fancyhdr      % 页眉页脚
- hyperref      % 超链接
- graphicx      % 图片
- booktabs      % 表格
- amsmath       % 数学公式
```

## 使用方法

### 1. 编译方式

**推荐使用 XeLaTeX 编译：**

```bash
xelatex report.tex
xelatex report.tex  # 第二次编译生成目录和交叉引用
```

**或使用 LaTeX 编辑器：**
- Overleaf: 上传 `report.tex`，选择 XeLaTeX 编译器
- TeXstudio: 设置默认编译器为 XeLaTeX
- VS Code + LaTeX Workshop: 配置使用 XeLaTeX

### 2. 修改内容

**基本信息修改（第50-60行）：**
```latex
\textbf{课程编号：}B2512009H \\
\textbf{课程名称：}自然语言处理（研讨课） \\
\textbf{任课教师：}赵阳
```

**标题修改（第80行）：**
```latex
{\LARGE \textbf{\color{titlecolor}期末大作业报告}}
```

**正文内容：**
- 直接修改各个 `\section{}` 和 `\subsection{}` 的内容
- 添加图片：使用 `\includegraphics{}`
- 添加表格：使用 `tabular` 环境
- 添加公式：使用 `equation` 或 `align` 环境

### 3. 自定义样式

**修改颜色主题（第26-28行）：**
```latex
\definecolor{titlecolor}{RGB}{0, 51, 102}      % 标题颜色
\definecolor{sectioncolor}{RGB}{0, 102, 204}   % 章节颜色
\definecolor{grayline}{RGB}{150, 150, 150}     % 分割线颜色
```

**修改页边距（第4行）：**
```latex
\usepackage[a4paper, top=2cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}
```

## 模板预览

### 页眉设计
```
2025-2026 学年秋季学期    期末考试    本科生试题专用纸
─────────────────────────────────────────────────
```

### 标题页
```
        中国科学院大学              课程编号：B2512009H
                                    课程名称：自然语言处理（研讨课）
        试 题 专 用 纸              任课教师：赵阳
─────────────────────────────────────────────────
注意事项：
1. 考核方式：____________________
═════════════════════════════════════════════════

              期末大作业报告
```

### 内容结构
- 项目背景
  - 研究动机
  - 相关工作
- 方法
  - 数据预处理
  - 模型架构
  - 训练策略
- 实验设置
  - 数据集
  - 实验参数
- 实验结果
- 结论与展望

## 示例内容

模板包含完整的NLP文本分类项目示例，涵盖：
- 研究背景和动机
- 相关工作综述
- 方法论描述（BERT模型）
- 实验设置和参数
- 结果分析
- 结论和未来工作

可以直接基于此示例修改为自己的项目内容。

## 文件说明

- `report.tex` - LaTeX源文件
- `report.pdf` - 编译后的PDF文件
- `report.synctex.gz` - 同步文件（用于编辑器跳转）

## 常见问题

### Q1: 编译报错 "ctex not found"？
**解决方案**: 安装完整的TeX发行版（TeX Live 或 MiKTeX）

### Q2: 中文显示乱码？
**解决方案**: 确保使用 XeLaTeX 编译，不要使用 PDFLaTeX

### Q3: 如何添加参考文献？
**解决方案**: 在文档末尾添加：
```latex
\bibliographystyle{plain}
\bibliography{references}
```

### Q4: 如何调整行距？
**解决方案**: 使用 setspace 宏包：
```latex
\usepackage{setspace}
\setstretch{1.5}  % 1.5倍行距
```

## 技术支持

如遇到问题，可以：
1. 查看 LaTeX 官方文档
2. 访问 TeX Stack Exchange
3. 参考 ctex 宏包文档

## 许可

本模板可自由使用和修改，适用于学术报告和课程作业。
