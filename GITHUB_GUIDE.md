# GitHub 上传指南

## 方法一：命令行上传（推荐）

### 1. 初始化本地仓库

```bash
# 在项目根目录下执行
git init
```

### 2. 添加文件到暂存区

```bash
# 添加所有文件
git add .

# 或者选择性添加
git add README.md
git add 1/
git add TextClassification/
git add SequenceLabeling/
git add MachineTranslation/
git add LLM/
```

### 3. 提交到本地仓库

```bash
git commit -m "Initial commit: NLP课程作业集"
```

### 4. 在GitHub上创建远程仓库

1. 访问 https://github.com
2. 点击右上角 "+" → "New repository"
3. 填写仓库名称，例如：`nlp-course-assignments`
4. 选择 Public（公开）或 Private（私有）
5. **不要**勾选 "Initialize this repository with a README"
6. 点击 "Create repository"

### 5. 关联远程仓库

```bash
# 替换 YOUR_USERNAME 和 REPO_NAME
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 例如：
# git remote add origin https://github.com/zhangsan/nlp-course-assignments.git
```

### 6. 推送到GitHub

```bash
# 第一次推送
git branch -M main
git push -u origin main

# 之后的推送只需要
git push
```

---

## 方法二：GitHub Desktop（图形界面）

### 1. 下载安装 GitHub Desktop
- 访问 https://desktop.github.com/
- 下载并安装

### 2. 登录GitHub账号
- 打开 GitHub Desktop
- File → Options → Accounts → Sign in

### 3. 添加本地仓库
- File → Add local repository
- 选择你的项目文件夹
- 如果提示"未初始化"，点击 "Create a repository"

### 4. 提交更改
- 在左侧看到所有更改的文件
- 在底部填写 Commit message：`Initial commit: NLP课程作业集`
- 点击 "Commit to main"

### 5. 发布到GitHub
- 点击顶部 "Publish repository"
- 填写仓库名称和描述
- 选择 Public 或 Private
- 点击 "Publish repository"

---

## 常见问题

### Q1: 文件太大无法上传怎么办？

**方案1: 使用 .gitignore（已配置）**
```bash
# 确保大文件被忽略
git rm --cached -r save_model/
git rm --cached -r data/
git commit -m "Remove large files"
```

**方案2: 使用 Git LFS（大文件存储）**
```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS"
```

### Q2: 推送时要求输入用户名密码？

GitHub已不支持密码认证，需要使用 Personal Access Token：

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成并复制 token
5. 推送时用 token 替代密码

**或者使用 SSH：**
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到 GitHub
# 复制 ~/.ssh/id_ed25519.pub 内容
# 访问 https://github.com/settings/keys
# 点击 "New SSH key" 并粘贴

# 修改远程地址为 SSH
git remote set-url origin git@github.com:YOUR_USERNAME/REPO_NAME.git
```

### Q3: 如何更新已上传的代码？

```bash
# 1. 修改文件后
git add .

# 2. 提交更改
git commit -m "更新说明"

# 3. 推送到GitHub
git push
```

### Q4: 如何查看当前状态？

```bash
# 查看文件状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v
```

---

## 推荐的提交信息规范

```bash
# 初始提交
git commit -m "Initial commit: NLP课程作业集"

# 添加新功能
git commit -m "feat: 添加文本分类模型"

# 修复bug
git commit -m "fix: 修复序列标注模型的梯度问题"

# 更新文档
git commit -m "docs: 更新README和使用说明"

# 优化代码
git commit -m "refactor: 重构数据处理模块"
```

---

## 完整示例流程

```bash
# 1. 初始化
cd /path/to/your/project
git init

# 2. 添加文件
git add .

# 3. 提交
git commit -m "Initial commit: NLP课程作业集"

# 4. 关联远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/nlp-course-assignments.git

# 5. 推送
git branch -M main
git push -u origin main
```

---

## 检查清单

上传前请确认：

- [ ] 已删除或忽略大文件（模型权重、数据集）
- [ ] 已删除敏感信息（API密钥、密码）
- [ ] README.md 文件完整
- [ ] .gitignore 配置正确
- [ ] 代码可以正常运行
- [ ] 已添加必要的说明文档

---

## 需要帮助？

如果遇到问题，可以：
1. 查看 Git 官方文档：https://git-scm.com/doc
2. 查看 GitHub 帮助：https://docs.github.com/
3. 搜索错误信息
