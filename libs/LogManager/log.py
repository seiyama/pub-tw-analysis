import pathlib
from os import path
import datetime
import logging

def getMyLogger(name, strLevel = 'INFO'):
  """

  ログをコンソール、ファイルに出力

  Args:
    name (str)    : __name__（実行しているプログラムのモジュール名）
    strLevel (str): ロギングレベル

  Returns:
    class: logger

  Raises:
    -

  Examples:
      logger = getMyLogger(__name__)
      logger.debug('デバッグ')

  """
  level_dict = {
    'CRITICAL':logging.CRITICAL,
    'ERROR':logging.ERROR,
    'WARNING':logging.WARNING,
    'INFO':logging.INFO,
    'DEBUG':logging.DEBUG
  }
  bc = logging.INFO if strLevel not in level_dict else level_dict[strLevel]
  logging.basicConfig(level=bc)
  logger = logging.getLogger(name)
  # レベルをDEBUG以上に設定
  logger.setLevel(logging.DEBUG)
  # ログの出力先
  today = "{0:%Y%m%d}".format(datetime.datetime.now())
  # log_file = path.join("..","logs",today+'.log')
  log_file = u'logs/'+today+u'.log'
  if not path.exists(log_file):
    with open(log_file, 'w') as f:
      f.write('')
      f.close()
  handler = logging.FileHandler(log_file)
  # ハンドラの対象のレベルを設定(DEBUG以上)
  handler.setLevel(logging.DEBUG)
  # フォーマットを指定
  formatter = logging.Formatter(
      '%(levelname)-9s  %(asctime)s  [%(name)s] %(message)s')
  handler.setFormatter(formatter)
  # 設定したハンドラをloggerに適用
  logger.addHandler(handler)
  return logger
