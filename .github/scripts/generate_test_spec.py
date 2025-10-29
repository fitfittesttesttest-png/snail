import os
import sys
from google import genai
from google.genai import types

# 設定値
PROJECT_ID = "fit-cloudgemini" 
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-exp"
TEMPERATURE = 0
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 8192
RESPONSE_MODALITIES = ["TEXT"]
TEST_SPEC_HEADER = "ケースID, テスト大項目,テスト中項目,小項目,テスト内容,期待結果,実施日時,ステータス"
SYSTEM_INSTRUCTION = f"""システムの設計書が渡されるので、設計に対してテストを作成してください。
テスト仕様書をカンマ区切りで出力してください。
テスト仕様書以外の情報は不要です。
ヘッダ行は以下のようにしてください。
{TEST_SPEC_HEADER}
実施日、ステータスは空欄で構いません。
"""

SAFETY_SETTINGS = [types.SafetySetting(
    category="HARM_CATEGORY_HATE_SPEECH",
    threshold="OFF"
  ),types.SafetySetting(
    category="HARM_CATEGORY_DANGEROUS_CONTENT",
    threshold="OFF"
  ),types.SafetySetting(
    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
    threshold="OFF"
  ),types.SafetySetting(
    category="HARM_CATEGORY_HARASSMENT",
    threshold="OFF"
  )
]

def generate_test_spec(document: str) -> str:
    """設計ドキュメントからテスト仕様書を生成する。"""
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
    except Exception as e:
        print(f"クライアントの初期化中にエラーが発生しました: {e}", file=sys.stderr)
        # 実行環境によっては PROJECT_ID の設定が必要な場合があります

    # 以前の "system" ロールによる content は削除済み
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=document)]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_modalities=RESPONSE_MODALITIES,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=SYSTEM_INSTRUCTION # Configuration オブジェクトに追加
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
        # 以前ここにあった system_instruction=... の引数は削除
    )
    return response.text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python generate_test_spec.py <設計書ファイル> [設計書ファイル2...]")
        sys.exit(1)

    input_file_paths = sys.argv[1:]
    for input_file_path in input_file_paths:
      try:
        with open(input_file_path, "r", encoding="utf-8") as f:
          design_document = f.read()
      except FileNotFoundError:
        print(f"エラー: ファイル '{input_file_path}' が見つかりません。", file=sys.stderr)
        continue
      except UnicodeDecodeError:
        print(f"エラー: ファイル '{input_file_path}' のエンコーディングがUTF-8ではありません。", file=sys.stderr)
        continue
        
      print(f"ファイル '{input_file_path}' のテスト仕様書を生成中です...")
      
      try:
          test_spec_content = generate_test_spec(design_document)
      except Exception as e:
          print(f"API呼び出し中にエラーが発生しました: {e}", file=sys.stderr)
          continue
          
      output_file_path = input_file_path.replace("docs", "tests").replace(".md", ".csv")
      
      # .mdがファイル名に含まれていない場合のフォールバック（例: .txtファイルなど）
      if not output_file_path.endswith(".csv"):
          output_file_path = os.path.splitext(input_file_path)[0].replace("docs", "tests") + ".csv"
          
      os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
      
      try:
          with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(test_spec_content)
          print(f"テスト仕様書を '{output_file_path}' に保存しました。")
      except Exception as e:
          print(f"ファイルへの書き込み中にエラーが発生しました: {e}", file=sys.stderr)
          
      print("-" * 30)
