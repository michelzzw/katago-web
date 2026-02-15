"""
KataGo Analysis Engine Manager
管理 KataGo 子进程，通过 Analysis JSON 协议通信
"""

import subprocess
import threading
import json
import os
import sys
import time
import queue
import logging

logger = logging.getLogger(__name__)


class KataGoEngine:
    """KataGo 分析引擎封装"""

    def __init__(self, katago_path, model_path, config_path, max_wait=300):
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self.max_wait = max_wait

        self.process = None
        self.lock = threading.Lock()
        self.response_queues = {}  # id -> Queue
        self.reader_thread = None
        self.running = False
        self.ready = False
        self.query_counter = 0

    def start(self):
        """启动 KataGo 分析引擎进程"""
        if self.process and self.process.poll() is None:
            logger.warning("KataGo 已在运行")
            return True

        cmd = [
            self.katago_path,
            "analysis",
            "-model", self.model_path,
            "-config", self.config_path,
        ]

        logger.info(f"启动 KataGo: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(self.katago_path) or ".",
            )
        except FileNotFoundError:
            logger.error(f"找不到 KataGo 可执行文件: {self.katago_path}")
            return False
        except Exception as e:
            logger.error(f"启动 KataGo 失败: {e}")
            return False

        self.running = True

        # 启动 stdout 读取线程
        self.reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader_thread.start()

        # 启动 stderr 读取线程（用于日志）
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

        # 等待引擎就绪 - 发送一个测试查询
        logger.info("等待 KataGo 引擎就绪（首次运行 OpenCL tuning 可能需要数分钟）...")
        # 等待进程稳定，首次 OpenCL tuning 需要较长时间
        for i in range(60):
            time.sleep(2)
            if self.process.poll() is not None:
                stderr_out = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
                logger.error(f"KataGo 进程已退出 (code={self.process.returncode})")
                if stderr_out:
                    logger.error(f"stderr: {stderr_out[:2000]}")
                return False
            # 尝试发送测试查询
            if i >= 2:  # 至少等 4 秒再试
                break
        
        logger.info("发送测试查询...")

        # 发一个简单查询来测试引擎是否就绪
        test_result = self.query(
            moves=[],
            board_size=9,
            komi=7.5,
            max_visits=1,
            query_id="__startup_test__"
        )

        if test_result is not None:
            self.ready = True
            logger.info("KataGo 引擎已就绪！")
            return True
        else:
            logger.error("KataGo 引擎启动超时")
            self.stop()
            return False

    def stop(self):
        """停止 KataGo 进程"""
        self.running = False
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            logger.info("KataGo 已停止")
        self.ready = False

    def _read_stdout(self):
        """从 stdout 读取 KataGo 分析结果（JSON 逐行）"""
        try:
            for line in self.process.stdout:
                if not self.running:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    response = json.loads(line)
                    query_id = response.get("id", "")
                    if query_id in self.response_queues:
                        self.response_queues[query_id].put(response)
                    else:
                        logger.debug(f"收到未知ID的响应: {query_id}")
                except json.JSONDecodeError:
                    logger.debug(f"非JSON输出: {line}")
        except Exception as e:
            if self.running:
                logger.error(f"读取 stdout 出错: {e}")

    def _read_stderr(self):
        """读取 stderr 日志"""
        try:
            for line in self.process.stderr:
                if not self.running:
                    break
                line = line.decode("utf-8").strip()
                if line:
                    logger.debug(f"[KataGo] {line}")
        except Exception:
            pass

    def _next_id(self):
        """生成唯一查询ID"""
        with self.lock:
            self.query_counter += 1
            return f"q_{self.query_counter}"

    def query(self, moves, board_size=19, komi=7.5, max_visits=3000,
              rules="chinese", query_id=None, analyze_turns=None,
              initial_stones=None, initial_player=None,
              include_ownership=False, include_policy=False):
        """
        发送分析查询到 KataGo

        Args:
            moves: 棋步列表 [["B", "D4"], ["W", "Q16"], ...]
            board_size: 棋盘大小
            komi: 贴目
            max_visits: 最大搜索次数
            rules: 规则 (chinese/japanese/korean 等)
            query_id: 自定义查询ID
            initial_stones: 初始棋子 [["B", "D4"], ...]
            include_ownership: 是否包含目数分析
            include_policy: 是否包含策略输出

        Returns:
            分析结果字典，失败返回 None
        """
        if not self.process or self.process.poll() is not None:
            logger.error("KataGo 进程未运行")
            return None

        if query_id is None:
            query_id = self._next_id()

        query_obj = {
            "id": query_id,
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "maxVisits": max_visits,
        }

        if initial_stones:
            query_obj["initialStones"] = initial_stones

        if initial_player:
            query_obj["initialPlayer"] = initial_player

        if analyze_turns is not None:
            query_obj["analyzeTurns"] = analyze_turns

        if include_ownership:
            query_obj["includeOwnership"] = True

        if include_policy:
            query_obj["includePolicy"] = True

        # 创建响应队列
        resp_queue = queue.Queue()
        self.response_queues[query_id] = resp_queue

        # 发送查询
        query_json = json.dumps(query_obj) + "\n"
        try:
            self.process.stdin.write(query_json.encode("utf-8"))
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"发送查询失败: {e}")
            del self.response_queues[query_id]
            return None

        # 等待响应
        try:
            response = resp_queue.get(timeout=self.max_wait)
            return response
        except queue.Empty:
            logger.warning(f"查询 {query_id} 超时")
            return None
        finally:
            del self.response_queues[query_id]

    def analyze_position(self, moves, board_size=19, komi=7.5,
                         max_visits=3000, include_ownership=False,
                         initial_stones=None, initial_player=None):
        """
        分析当前局面，返回推荐走法和胜率

        Returns:
            {
                "currentPlayer": "B" or "W",
                "winrate": float,       # 黑棋胜率
                "scoreLead": float,     # 黑棋领先目数
                "moves": [              # 推荐走法列表
                    {
                        "move": "D4",
                        "visits": 1234,
                        "winrate": 0.55,
                        "scoreLead": 2.3,
                        "order": 0,
                        "pv": ["D4", "Q16", ...]
                    },
                    ...
                ],
                "ownership": [...] or None  # 目数归属（可选）
            }
        """
        result = self.query(
            moves=moves,
            board_size=board_size,
            komi=komi,
            max_visits=max_visits,
            include_ownership=include_ownership,
            initial_stones=initial_stones,
            initial_player=initial_player,
        )

        if result is None:
            return None

        if "error" in result:
            logger.error(f"KataGo 分析错误: {result['error']}")
            return None

        # 解析结果
        move_infos = result.get("moveInfos", [])
        root_info = result.get("rootInfo", {})

        parsed = {
            "currentPlayer": root_info.get("currentPlayer", "B"),
            "winrate": root_info.get("winrate", 0.5),
            "scoreLead": root_info.get("scoreLead", 0.0),
            "visits": root_info.get("visits", 0),
            "moves": [],
        }

        for mi in move_infos:
            parsed["moves"].append({
                "move": mi.get("move", "pass"),
                "visits": mi.get("visits", 0),
                "winrate": mi.get("winrate", 0.5),
                "scoreLead": mi.get("scoreLead", 0.0),
                "order": mi.get("order", 0),
                "pv": mi.get("pv", []),
                "prior": mi.get("prior", 0.0),
            })

        if include_ownership and "ownership" in result:
            parsed["ownership"] = result["ownership"]

        return parsed

    def is_running(self):
        """检查引擎是否在运行"""
        return self.process is not None and self.process.poll() is None and self.ready
