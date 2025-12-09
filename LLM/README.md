# 大语言模型API调用实践

## 项目简介

本项目实践了大语言模型API的调用和测试，包括请求构造、身份验证和错误处理。

## 作业内容

- API接口调用测试
- HTTP请求头配置（User-Agent伪装）
- 错误处理（403绕过尝试）
- 训练数据格式（JSONL）准备

## 文件说明

- `test.py`: API调用测试脚本
- `capture.py`: 数据捕获工具
- `text.py`: 文本处理工具
- `train.jsonl`: 训练数据集
- `test.jsonl`: 测试数据集
- `results.txt`: 测试结果

## 运行方式

```bash
python test.py
```

注意：需要配置有效的API密钥。
