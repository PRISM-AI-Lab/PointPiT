# PointPiT Project Website

这是 PointPiT 的独立项目主页目录。

## 本地预览

在整个工作区根目录运行：

```bash
python -m http.server 8000 --directory web
```

访问 `http://localhost:8000`。

## 发布

若使用独立的 GitHub Pages 仓库，将 `web/` 中的全部内容放在该仓库根目录，
然后在仓库 Settings → Pages 中选择从默认分支根目录发布即可。

发布前请在 `index.html` 中更新作者、论文链接、会议状态和 BibTeX 信息。
