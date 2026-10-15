"""安装阶段准备官方参考Tokenizer；服务处理请求时不下载分词表。"""
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cache = root / 'data' / 'tokenizers'
cache.mkdir(parents=True, exist_ok=True)
os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
import tiktoken

# tiktoken包内置官方来源和SHA256校验；失败时不退回字符计数。
encoding = tiktoken.get_encoding('cl100k_base')
assert encoding.encode('hello world') == [15339, 1917]
print('Tokenizer ready: cl100k_base reference estimate; provider usage may differ.')
