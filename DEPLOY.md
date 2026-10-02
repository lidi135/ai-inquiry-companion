# AI 问诊陪练助手 —— Streamlit Cloud 部署指引

## 背景
本项目已托管在 GitHub：
- 仓库：https://github.com/lidi135/ai-inquiry-companion
- 主分支：main
- 入口文件：src/app.py

## 部署平台
[Streamlit Community Cloud](https://share.streamlit.io) —— 免费、稳定、随 commit 自动更新。

## 一次性设置（首次登录后约 3 分钟）

1. 用 GitHub 账号登录 share.streamlit.io。
2. 第一次会弹 **「Set up your account」** 表单，需要补四项信息：
   - **First name / Last name**：填你的姓名（任填）
   - **Primary email**：自动用 GitHub 邮箱，可保留
   - **What's your functional area?**：下拉选 `Education` 或 `Healthcare`（与项目场景最贴近）
   - **What stage of app development are you at?**：选 `Idea` 或 `Prototype`
   - **Country or region**：默认 `China`，不动
   - 点底部蓝色 **Continue** 按钮。

## 创建应用

1. 进入控制台后，点 **New app**（或 Create app）
3. 表单填写：
   - **Repository**：`lidi135/ai-inquiry-companion`
   - **Branch**：`main`
   - **Main file path**：`src/app.py`
   - **App URL**：自定义子域，比如 `ai-inquiry-companion`（生成 `https://ai-inquiry-companion.streamlit.app`）

4. 点 **Advanced settings** → **Secrets** 标签 → 粘贴：
   ```toml
   DEEPSEEK_API_KEY = "sk-你的真实key"
   ```
   （任何其他需要注入的密钥同理配置）
5. 点 **Deploy!**。首次会拉依赖，1-3 分钟后状态变绿，得到公网 URL。

## 当前公网地址（已验证可用）

```
https://ai-inquiry-companion-lpqczxpzvtmerht4oddokv.streamlit.app/
```

复制上述链接给任何人都能用。**注意**：每个 Streamlit Cloud 应用的实际 URL 后缀是部署时随机生成的短哈希，请以你 share.streamlit.io 控制台「My apps」里显示的 URL 为准。

## 验证
- 打开 App URL，应能看到 Hero 头图与三 Tab 界面。
- 在侧边栏选个模型，发一句问诊，得到 AI 病人回复 → 说明“API 密钥 + 部署”都成功。
- 公网匿名访问测试（从本人电脑）已通过 `Invoke-WebRequest` 返回 `STATUS=200`。

## 常见问题
- **访问慢/休眠**：免费版空闲会休眠，第一次访问会自动唤醒（3-5 秒）。
- **依赖报错**：检查仓库根目录 requirements.txt 是否同步了 src/requirements.txt。
- **Secrets 不生效**：必须是 TOML 格式（key = "value"），保存后点 ⋮ → Reboot。
- **代码更新**：devtools > 项目改后 `git push` 到 main，Cloud 会 1-2 分钟内自动重建。