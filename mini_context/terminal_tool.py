"""
按《HelloAgents》第九章 9.5 节代码实现的 TerminalTool。
四层安全机制:命令白名单、工作目录沙箱、超时控制、输出大小限制。
"""
import os
import subprocess
from pathlib import Path
from typing import List

ALLOWED_COMMANDS = {
    "ls", "dir", "tree",
    "cat", "head", "tail", "less", "more",
    "find", "grep", "egrep", "fgrep",
    "wc", "sort", "uniq", "cut", "awk", "sed",
    "pwd", "cd",
    "file", "stat", "du", "df",
    "echo", "which", "whereis",
}


class TerminalTool:
    def __init__(self, workspace: str, timeout: int = 30, max_output_size: int = 10 * 1024 * 1024,
                 allow_cd: bool = True):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.current_dir = self.workspace
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd

    def run(self, params: dict) -> str:
        command = params.get("command", "").strip()
        if not command:
            return "❌ 未提供命令"

        parts = command.split()
        base_cmd = parts[0]

        if base_cmd not in ALLOWED_COMMANDS:
            allowed_preview = ", ".join(sorted(ALLOWED_COMMANDS)[:10]) + "..."
            return f"❌ 不允许的命令: {base_cmd}\n允许的命令: {allowed_preview}"

        if base_cmd == "cd":
            return self._handle_cd(parts)

        return self._execute_command(command)

    # ---------- 第一层: 命令白名单(在run里已做,这里是第三/四层:超时+输出限制) ----------
    def _execute_command(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy(),
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            if len(output) > self.max_output_size:
                output = output[: self.max_output_size]
                output += f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"

            if result.returncode != 0:
                output = f"⚠️ 命令返回码: {result.returncode}\n\n{output}"

            return output if output else "✅ 命令执行成功（无输出）"

        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时（超过 {self.timeout} 秒）"
        except Exception as e:
            return f"❌ 命令执行失败: {e}"

    # ---------- 第二层: 工作目录沙箱 ----------
    def _handle_cd(self, parts: List[str]) -> str:
        if not self.allow_cd:
            return "❌ cd 命令已禁用"

        if len(parts) < 2:
            return f"当前目录: {self.current_dir}"

        target_dir = parts[1]

        if target_dir == "..":
            new_dir = self.current_dir.parent
        elif target_dir == ".":
            new_dir = self.current_dir
        elif target_dir == "~":
            new_dir = self.workspace
        else:
            new_dir = (self.current_dir / target_dir).resolve()

        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {new_dir}"

        if not new_dir.exists():
            return f"❌ 目录不存在: {new_dir}"
        if not new_dir.is_dir():
            return f"❌ 不是目录: {new_dir}"

        self.current_dir = new_dir
        return f"✅ 切换到目录: {self.current_dir}"
