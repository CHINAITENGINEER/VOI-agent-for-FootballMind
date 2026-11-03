# 协作与模型环境

## 协作方式

从 `main` 新建 `feature/<name>` 或 `fix/<name>` 分支，修改完成后提交 Pull Request。不要直接覆盖冻结包、原始样本或实验结果。当前仓库未自动设置分支保护；仓库管理员可按团队习惯设置。

建议 PR 说明问题、最终行为、测试命令，以及是否改变模型/数据版本。尚无授权的开源许可证，不擅自给第三方模型或新闻语料补许可证。

## 当前入口

- `Start-FootballMind.cmd` / `start_footballmind_defense.py` 默认打开在线页面。
- `python start_footballmind_defense.py --page defense` 打开可离线使用的冻结研究页面。
- `python -m demo --host 127.0.0.1 --port 8766` 以前台方式启动服务，便于看日志与停止。
- `start_footballmind.py` 为旧 human trial 入口，不作为当前演示首选。

## 统一配置与在线编码依赖

当前主入口为 `python -m footballmind doctor|serve|experiment`，完整说明见 [研究底座使用说明](development/research-foundation.md)。配置优先级：`--config` > `FOOTBALLMIND_CONFIG` > 当前工作区 `config/workspace.local.json` > 可迁移默认目录。配置与模型元数据无秘密字段；密钥仍只保留在环境变量中。

`demo/live_postread_model.py` 中当前冻结编码器版本为：

```text
Qwen/Qwen3-Embedding-0.6B
revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

原开发机快照目录（仅作为本机配置示例）：

```text
E:/FootballMind/models/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

原开发机缓存目录：`E:/FootballMind/embeddings/live-postread-qwen-v1`。

现在将这些路径写进 `config/workspace.local.json` 的 `encoder_dir`、`cache_dir` 和 `release_dir`。普通服务器、启动器与诊断使用同一配置。没有本机配置时默认使用工作区内的 `var/models`、`var/cache` 与 `var/runs`，不探测或偷偷回退到另一块磁盘。配置支持迁移路径，当前模型算法版本与 SHA 仍固定；升级模型需独立实现和验证。

## 冻结包与大数据

仓库保留 `var/releases/footballmind-defense-20261025-rc1/manifest.json` 引用的全部364个文件及manifest，包含f/g权重、小型历史演示候选池和编码缓存。`.gitattributes`禁止Git改写这些文件换行，确保原SHA可复核。新的实验必须输出到新目录。

未纳入Git：完整新闻库、采集数据库、全部实验产物、第三方编码器权重、运行日志及私密凭据。`var/`除指定冻结包外均被忽略。共享大数据时另行提供数据版本、SHA与存放路径。

若要扩展采集平台，另安装 `football_spider/requirements.txt`；数据库、调度器、浏览器测试等属于可选模块，不是离线演示的前置依赖。

## 安全和实验边界

不要提交API Key、含口令的代理URL、用户浏览器资料或数据库。仓库中的认证与URL安全测试使用测试占位值，不用于真实访问。

保持比赛级分组、来源时间证据和冻结f绑定。新增模型采用新目录、新模型版本与新SHA；已有结论不因换模型而自动继承。线上运行失败保留失败记录，不复制before作为有效after，也不生成虚假的零收益标签。
