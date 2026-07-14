"""
按《HelloAgents》第九章 9.4 节代码实现的 NoteTool。
七个核心操作:create/read/update/search/list/summary/delete。
存储格式:Markdown正文 + YAML前置元数据,配合一个notes_index.json做快速索引。
"""
import os
import json
import yaml
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


class NoteTool:
    def __init__(self, workspace: str):
        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)
        self.index_path = os.path.join(self.workspace, "notes_index.json")
        self.index: Dict[str, Dict] = {}
        self._load_index()

    # ---------- 统一入口(书中用run({"action": ...})调用,这里对齐同样的调用习惯) ----------
    def run(self, params: Dict[str, Any]):
        action = params.get("action")
        if action == "create":
            return self._create_note(params["title"], params["content"],
                                       params.get("note_type", "general"), params.get("tags"))
        elif action == "read":
            return self._read_note(params["note_id"])
        elif action == "update":
            return self._update_note(params["note_id"], params.get("title"), params.get("content"),
                                       params.get("note_type"), params.get("tags"))
        elif action == "search":
            return self._search_notes(params["query"], params.get("limit", 10),
                                        params.get("note_type"), params.get("tags"))
        elif action == "list":
            return self._list_notes(params.get("note_type"), params.get("tags"), params.get("limit", 20))
        elif action == "summary":
            return self._summary()
        elif action == "delete":
            return self._delete_note(params["note_id"])
        else:
            raise ValueError(f"未知action: {action}")

    # ---------- create ----------
    def _create_note(self, title: str, content: str, note_type: str = "general",
                      tags: Optional[List[str]] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_id = f"note_{timestamp}_{len(self.index)}"

        metadata = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        md_content = self._build_markdown(metadata, content)
        file_path = os.path.join(self.workspace, f"{note_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        metadata["file_path"] = file_path
        self.index[note_id] = metadata
        self._save_index()
        return note_id

    def _build_markdown(self, metadata: Dict, content: str) -> str:
        yaml_header = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
        return f"---\n{yaml_header}---\n\n{content}"

    # ---------- read ----------
    def _read_note(self, note_id: str) -> Dict:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        file_path = self.index[note_id]["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        metadata, content = self._parse_markdown(raw_content)
        return {"metadata": metadata, "content": content}

    def _parse_markdown(self, raw_content: str) -> Tuple[Dict, str]:
        parts = raw_content.split("---\n", 2)
        if len(parts) >= 3:
            yaml_str = parts[1]
            content = parts[2].strip()
            metadata = yaml.safe_load(yaml_str)
        else:
            metadata = {}
            content = raw_content.strip()
        return metadata, content

    # ---------- update ----------
    def _update_note(self, note_id: str, title: Optional[str] = None, content: Optional[str] = None,
                      note_type: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        note = self._read_note(note_id)
        metadata = note["metadata"]
        old_content = note["content"]

        if title:
            metadata["title"] = title
        if note_type:
            metadata["type"] = note_type
        if tags is not None:
            metadata["tags"] = tags
        if content is not None:
            old_content = content

        metadata["updated_at"] = datetime.now().isoformat()

        md_content = self._build_markdown(metadata, old_content)
        file_path = metadata["file_path"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.index[note_id] = metadata
        self._save_index()
        return f"✅ 笔记已更新: {metadata['title']}"

    # ---------- search ----------
    def _search_notes(self, query: str, limit: int = 10, note_type: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> List[Dict]:
        results = []
        query_lower = query.lower()

        for note_id, metadata in self.index.items():
            if note_type and metadata.get("type") != note_type:
                continue
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue
            try:
                note = self._read_note(note_id)
                content = note["content"]
                title = metadata.get("title", "")
                if query_lower in title.lower() or query_lower in content.lower():
                    results.append({
                        "note_id": note_id,
                        "title": title,
                        "type": metadata.get("type"),
                        "tags": metadata.get("tags", []),
                        "content": content,
                        "updated_at": metadata.get("updated_at"),
                    })
            except Exception as e:
                print(f"[WARNING] 读取笔记 {note_id} 失败: {e}")
                continue

        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results[:limit]

    # ---------- list ----------
    def _list_notes(self, note_type: Optional[str] = None, tags: Optional[List[str]] = None,
                     limit: int = 20) -> List[Dict]:
        results = []
        for note_id, metadata in self.index.items():
            if note_type and metadata.get("type") != note_type:
                continue
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue
            results.append(metadata)
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    # ---------- summary ----------
    def _summary(self) -> Dict[str, Any]:
        total_count = len(self.index)
        type_counts = {}
        for metadata in self.index.values():
            note_type = metadata.get("type", "general")
            type_counts[note_type] = type_counts.get(note_type, 0) + 1

        recent_notes = sorted(self.index.values(), key=lambda x: x.get("updated_at", ""), reverse=True)[:5]

        return {
            "total_notes": total_count,
            "type_distribution": type_counts,
            "recent_notes": [
                {"id": n["id"], "title": n.get("title", ""), "type": n.get("type"), "updated_at": n.get("updated_at")}
                for n in recent_notes
            ],
        }

    # ---------- delete ----------
    def _delete_note(self, note_id: str) -> str:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        file_path = self.index[note_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
        title = self.index[note_id].get("title", note_id)
        del self.index[note_id]
        self._save_index()
        return f"✅ 笔记已删除: {title}"

    # ---------- 索引持久化 ----------
    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
