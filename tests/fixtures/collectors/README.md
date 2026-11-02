# A 的来源样本交接

真实样本由 `tests/platform/collectors/export_evidence.py` 从既有归档提取；原档案不修改、不重新请求。每个 payload 配套 metadata，包含原观测 UUID、真实 UTC observed_at、HTTP status、原内容哈希、共享样本哈希、脱敏/转换方式。

`sample_kind=real_capture_redacted_copy` 不表示新的观测。这些样本仅用于测试与字段核对，**不要导入生产库当作刚采集的数据**。重复样本必须有不同的真实 observation_id，不能复制后编造新时间。

HTML 仅共享去除 script/style/属性后的可见文本，避免传播 challenge token；原始正文仍在旧平台私有 raw 档案。缺正文的真实网络失败保存零字节 `.bin`，由 metadata 的 error_class 解释。

`synthetic_*` 文件一律是合成测试夹具，不是联网数据。无 key 来源没有成功业务响应时，保持缺口，不造真响应。

总目录见 manifest.json；来源字段/周期/时间语义见根 docs/sources/。
