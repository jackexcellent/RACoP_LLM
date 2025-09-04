# 改進的 PsyMix CoP 導向訓練系統 - 完整版本
import os
import torch
import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
import re

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

#################################################################################
# 改進的 CoP 導向資料增強
#################################################################################

class ImprovedCoPDataAugmentation:
    """確保生成完整的治療師回應"""

    def __init__(self):
        self.response_starters = {
            'cbt': [
                "I can see from your thoughts that",
                "Based on what you're thinking about this situation",
                "Looking at the pattern of your thoughts"
            ],
            'pct': [
                "I sense the deep emotions you're experiencing",
                "Your feelings are telling us something important",
                "I'm hearing the emotional weight of"
            ],
            'sfbt': [
                "Focusing on your strengths and goals",
                "Let's build on what's already working",
                "I notice you have clear aspirations"
            ]
        }

        self.validation_phrases = {
            'cbt': [
                "It's understandable that you would think this way given your experiences.",
                "These thought patterns often develop as ways to protect ourselves.",
                "Many people struggle with similar thoughts in these situations."
            ],
            'pct': [
                "Your feelings are completely valid and worth exploring.",
                "What you're experiencing makes sense in the context of your life.",
                "I appreciate your courage in sharing these emotions."
            ],
            'sfbt': [
                "Your desire for change shows real strength.",
                "You've already taken the first step by recognizing what you want.",
                "I can see you have important resources to draw upon."
            ]
        }

        self.exploration_questions = {
            'cbt': [
                "What evidence do you have for and against this thought?",
                "How would you advise a friend who had this thought?",
                "What's a more balanced way to look at this situation?",
                "What would it mean if this thought were true? And if it weren't?"
            ],
            'pct': [
                "What does this experience tell you about what matters most to you?",
                "How does this feeling connect to your sense of who you are?",
                "What would it be like to fully accept these emotions?",
                "What do you need most in this moment?"
            ],
            'sfbt': [
                "When have you successfully handled something similar?",
                "What would be different if you achieved this goal?",
                "On a scale of 1-10, where are you now and where would you like to be?",
                "What's the smallest step you could take today?"
            ]
        }

        self.therapeutic_techniques = {
            'cbt': [
                "Let's try a thought record exercise. Write down the situation, your automatic thoughts, the emotions you felt, and then we can work on finding more balanced alternatives.",
                "We can use cognitive restructuring to examine these beliefs more closely and develop healthier thinking patterns.",
                "It might help to identify the cognitive distortions at play here - perhaps all-or-nothing thinking or catastrophizing."
            ],
            'pct': [
                "I invite you to sit with these feelings without judgment, simply observing what arises.",
                "Let's explore what your authentic self is trying to communicate through these emotions.",
                "Focus on your present experience - what are you noticing in your body right now?"
            ],
            'sfbt': [
                "Let's imagine your preferred future - what would be happening differently?",
                "We can use scaling questions to track your progress toward your goals.",
                "Tell me about a time when this problem wasn't as severe - what was different?"
            ]
        }

    def create_full_response(self, cop_analysis, therapy_type, patient_input):
        """創建完整的治療師回應"""
        elements = self.extract_detailed_elements(cop_analysis)

        # 構建回應的各個部分
        response_parts = []

        # 1. 開場 - 承認並反映患者的主要關切
        starter = np.random.choice(self.response_starters[therapy_type])
        response_parts.append(f"{starter}, {self.reflect_main_concern(patient_input, elements, therapy_type)}")

        # 2. 驗證感受
        validation = np.random.choice(self.validation_phrases[therapy_type])
        response_parts.append(validation)

        # 3. 深入探討 - 引用 CoP 分析的具體內容
        if therapy_type == 'cbt':
            if 'event' in elements and 'cognition' in elements:
                response_parts.append(
                    f"When {elements['event']}, you find yourself thinking '{elements['cognition']}'. "
                    f"This thought seems to be causing you significant distress."
                )
        elif therapy_type == 'pct':
            if 'emotion' in elements:
                response_parts.append(
                    f"The {elements['emotion']} you're experiencing seems to be at the heart of your current struggle. "
                    f"Let's explore what this emotion might be telling us about your needs and values."
                )
        elif therapy_type == 'sfbt':
            if 'goal' in elements:
                response_parts.append(
                    f"Your goal to {elements['goal']} is clear and meaningful. "
                    f"Let's focus on the times when you've been closer to this goal."
                )

        # 4. 探索性問題
        question = np.random.choice(self.exploration_questions[therapy_type])
        response_parts.append(question)

        # 5. 提供具體技術或策略
        technique = np.random.choice(self.therapeutic_techniques[therapy_type])
        response_parts.append(technique)

        # 6. 結尾 - 支持和鼓勵
        response_parts.append(self.create_supportive_closing(therapy_type))

        # 組合成完整回應
        full_response = " ".join(response_parts)

        # 確保回應夠長且連貫
        if len(full_response) < 200:
            full_response += f" Remember, {self.get_encouragement(therapy_type)}"

        return full_response

    def reflect_main_concern(self, patient_input, elements, therapy_type):
        """反映患者的主要關切"""
        # 簡單的關鍵詞提取
        concerns = []
        if "anxious" in patient_input.lower() or "anxiety" in patient_input.lower():
            concerns.append("anxiety")
        if "depress" in patient_input.lower() or "sad" in patient_input.lower():
            concerns.append("low mood")
        if "relationship" in patient_input.lower():
            concerns.append("relationship challenges")
        if "work" in patient_input.lower() or "job" in patient_input.lower():
            concerns.append("work-related stress")

        if concerns:
            return f"you're dealing with {' and '.join(concerns)}"
        else:
            return "you're facing some challenging feelings and situations"

    def extract_detailed_elements(self, cop_analysis):
        """提取 CoP 分析的詳細元素"""
        elements = {}

        # 更詳細的模式匹配
        patterns = {
            'event': r'Event:\s*([^\n]+?)(?:\n|$)',
            'cognition': r'(?:Thought|Cognition):\s*([^\n]+?)(?:\n|$)',
            'feeling': r'Feeling:\s*([^\n]+?)(?:\n|$)',
            'belief': r'(?:Core )?Belief:\s*([^\n]+?)(?:\n|$)',
            'emotion': r'Emotion:\s*([^\n]+?)(?:\n|$)',
            'awareness': r'Awareness:\s*([^\n]+?)(?:\n|$)',
            'goal': r'Goal:\s*([^\n]+?)(?:\n|$)',
            'resource': r'(?:Strengths?|Resource):\s*([^\n]+?)(?:\n|$)',
            'action': r'(?:Next Steps?|Action):\s*([^\n]+?)(?:\n|$)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, cop_analysis, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value.lower() != 'n/a':
                    elements[key] = value

        return elements

    def create_supportive_closing(self, therapy_type):
        """創建支持性的結尾"""
        closings = {
            'cbt': [
                "Remember, thoughts are not facts, and we can work together to develop more helpful thinking patterns.",
                "With practice, you can learn to identify and challenge these unhelpful thoughts.",
                "Change takes time, but each small step in examining your thoughts is progress."
            ],
            'pct': [
                "I'm here to support you in this journey of self-discovery and growth.",
                "Your willingness to explore these feelings shows real courage.",
                "Trust in your own process - you have the wisdom within to find your way."
            ],
            'sfbt': [
                "Every positive change, no matter how small, is a step in the right direction.",
                "You have more resources and strengths than you might realize right now.",
                "Focus on progress, not perfection - celebrate each victory along the way."
            ]
        }
        return np.random.choice(closings[therapy_type])

    def get_encouragement(self, therapy_type):
        """獲取鼓勵語句"""
        encouragements = {
            'cbt': "changing thought patterns is a skill that improves with practice",
            'pct': "your journey toward authenticity and self-acceptance is valuable",
            'sfbt': "you have the strength and resources to create the changes you desire"
        }
        return encouragements[therapy_type]

#################################################################################
# 資料載入函數
#################################################################################

def load_data_flexible(path):
    """載入資料"""
    data_list = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data_list = data
            print(f"成功載入 {len(data_list)} 筆資料")
            return data_list
    except:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data_list.append(json.loads(line))
            print(f"成功載入 {len(data_list)} 筆資料")
            return data_list
        except Exception as e:
            print(f"載入失敗: {e}")
            return []

#################################################################################
# 改進的資料處理
#################################################################################

def create_improved_training_examples(data_list, max_samples=3000):
    """創建改進的訓練樣本，確保完整的回應生成"""
    augmenter = ImprovedCoPDataAugmentation()
    sft_examples = []
    therapy_types = ['cbt', 'pct', 'sfbt']

    # 過濾高品質資料
    filtered_data = []
    for data in data_list:
        if data.get('quality_score', 0.7) >= 0.7:
            filtered_data.append(data)

    print(f"過濾後剩餘 {len(filtered_data)} 筆資料")

    # 限制樣本數
    if len(filtered_data) > max_samples:
        filtered_data = filtered_data[:max_samples]

    for idx, data in enumerate(tqdm(filtered_data, desc="處理資料")):
        prompt = data.get('input', '').strip()
        cop_analyses = data.get('cop_analyses', {})

        if not prompt or not cop_analyses:
            continue

        # 為每種治療方法創建樣本
        for therapy_type in therapy_types:
            if therapy_type not in cop_analyses:
                continue

            analysis = cop_analyses[therapy_type]

            # 格式化 CoP 分析
            if therapy_type == 'cbt':
                analysis_text = f"""[CBT Analysis]
Event: {analysis.get('event', 'Current situation causing distress')}
Thought: {analysis.get('cognition', 'Automatic negative thoughts')}
Feeling: {analysis.get('behavior', 'Emotional and behavioral responses')}
Core Belief: {analysis.get('belief', 'Underlying beliefs about self/world')}"""

            elif therapy_type == 'pct':
                analysis_text = f"""[PCT Analysis]
Emotion: {analysis.get('emotion', 'Current emotional experience')}
Awareness: {analysis.get('self_awareness', 'Level of self-understanding')}
Authenticity: {analysis.get('authenticity', 'Connection to true self')}"""

            else:  # sfbt
                analysis_text = f"""[SFBT Analysis]
Goal: {analysis.get('goal', 'Desired future state')}
Strengths: {analysis.get('resource', 'Existing resources and abilities')}
Past Success: {analysis.get('exception', 'Times when problem was less severe')}
Next Steps: {analysis.get('action', 'Concrete actions toward goal')}"""

            # 創建完整的治療師回應
            therapeutic_response = augmenter.create_full_response(
                analysis_text,
                therapy_type,
                prompt
            )

            # 格式化訓練文本 - 使用更清晰的分隔
            formatted_text = f"""<|user|>
{prompt}
<|end|>
<|assistant|>
{analysis_text}

[Therapeutic Response]
{therapeutic_response}
<|end|>"""

            sft_examples.append({
                "text": formatted_text,
                "therapy_type": therapy_type,
                "quality_score": data.get('quality_score', 0.8)
            })

    print(f"創建了 {len(sft_examples)} 個改進的訓練樣本")
    np.random.shuffle(sft_examples)
    return sft_examples

#################################################################################
# 改進的生成函數
#################################################################################

def generate_improved_response(model, tokenizer, prompt, max_retries=3):
    """改進的生成函數，確保完整回應"""
    formatted_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    best_response = None
    best_cop = None

    for attempt in range(max_retries):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,  # 增加生成長度
                min_new_tokens=300,  # 確保最小長度
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 解析回應
        cop_part = ""
        therapy_response = ""

        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[1]
        else:
            response = full_response

        # 改進的解析邏輯
        if "[Therapeutic Response]" in response:
            parts = response.split("[Therapeutic Response]")
            cop_part = parts[0].strip()
            therapy_response = parts[1].strip() if len(parts) > 1 else ""
        elif "Analysis]" in response:
            # 找到分析結束的位置
            lines = response.split('\n')
            analysis_end = -1
            for i, line in enumerate(lines):
                if line.strip() == "" and i > 0 and any(marker in lines[i-1] for marker in ["Goal:", "Awareness:", "Belief:"]):
                    analysis_end = i
                    break

            if analysis_end > 0:
                cop_part = '\n'.join(lines[:analysis_end])
                therapy_response = '\n'.join(lines[analysis_end+1:])

        # 清理回應
        cop_part = cop_part.replace("<|end|>", "").strip()
        therapy_response = therapy_response.replace("<|end|>", "").strip()

        # 檢查回應品質
        if len(therapy_response) > 100:  # 確保回應夠長
            best_response = therapy_response
            best_cop = cop_part
            break
        elif len(therapy_response) > len(best_response or ""):
            best_response = therapy_response
            best_cop = cop_part

    # 如果仍然沒有好的回應，使用備用方案
    if not best_response or len(best_response) < 50:
        best_response = "I understand you're going through a difficult time. Based on what you've shared, it seems like you're experiencing some challenging emotions and situations. I'd like to explore this further with you. Can you tell me more about what specific aspects of this situation are most troubling for you? Understanding your perspective better will help us work together to find ways to address these concerns."

    return best_cop or "[Analysis not available]", best_response

#################################################################################
# 主程式
#################################################################################

# 載入資料
dataset_path = "/content/dataset.json"
print("\n=== 載入資料 ===")
raw_data = load_data_flexible(dataset_path)

if len(raw_data) == 0:
    print("使用範例資料")
    raw_data = [{
        "input": "Patient: I'm feeling very anxious about my upcoming presentation.",
        "cop_analyses": {
            "cbt": {
                "event": "Upcoming work presentation next week",
                "cognition": "I will mess up and everyone will judge me",
                "behavior": "Avoiding preparation, experiencing physical tension",
                "belief": "I must be perfect or I'm a failure"
            },
            "pct": {
                "emotion": "Intense anxiety, fear of judgment, self-doubt",
                "self_awareness": "Recognizes anxiety but feels trapped by perfectionist standards",
                "authenticity": "Struggling to be authentic due to fear"
            },
            "sfbt": {
                "goal": "Deliver a confident, well-prepared presentation",
                "resource": "Strong subject knowledge, previous successful presentations",
                "exception": "Felt confident in smaller group settings",
                "action": "Practice with trusted colleagues, prepare thoroughly"
            }
        },
        "therapist_response": "I understand you're feeling anxious.",
        "quality_score": 0.9
    }]

# 創建改進的訓練資料
print("\n=== 創建訓練資料 ===")
enhanced_data = create_improved_training_examples(raw_data, max_samples=3000)

# 建立資料集
train_dataset = Dataset.from_list(enhanced_data)

# 90/10 分割
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"\n訓練集: {len(train_dataset)} 筆")
print(f"驗證集: {len(eval_dataset)} 筆")

# 顯示範例
print("\n=== 訓練樣本範例 ===")
print(train_dataset[0]['text'][:800] + "...")
print("-" * 70)

#################################################################################
# 模型載入和配置
#################################################################################

model_name = "microsoft/Phi-3-mini-4k-instruct"

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"\n=== 載入模型: {model_name} ===")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True
)

# 設定 tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 準備模型
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

print(f"模型參數量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

#################################################################################
# LoRA 配置
#################################################################################

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#################################################################################
# 資料預處理函數
#################################################################################

def preprocess_function(examples):
    """預處理訓練資料"""
    model_inputs = tokenizer(
        examples["text"],
        max_length=768,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )

    # 創建 labels
    labels = []
    for input_ids in model_inputs["input_ids"]:
        label_ids = input_ids.copy()
        # Padding tokens 設為 -100
        for i in range(len(label_ids)):
            if label_ids[i] == tokenizer.pad_token_id:
                label_ids[i] = -100
        labels.append(label_ids)

    model_inputs["labels"] = labels
    return model_inputs

#################################################################################
# 訓練配置
#################################################################################

# 計算步數
steps_per_epoch = len(train_dataset) // 4  # batch_size * gradient_accumulation
total_steps = steps_per_epoch * 2  # 2 epochs

training_args = TrainingArguments(
    output_dir="./psymix-cop-focused",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=200,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    seed=42,
    logging_first_step=True,
)

print(f"\n預計訓練步數: {total_steps}")

#################################################################################
# 開始訓練
#################################################################################

print("\n=== 處理訓練資料 ===")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset"
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing eval dataset"
)

# 資料整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# 建立 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\n=== 開始 CoP 導向訓練 ===")
trainer.train()

# 儲存模型
trainer.save_model("./psymix-cop-focused-final")
tokenizer.save_pretrained("./psymix-cop-focused-final")
print("\n訓練完成！模型已儲存至 ./psymix-cop-focused-final")

#################################################################################
# 測試模型
#################################################################################

print("\n=== 測試改進的生成 ===")
model.eval()

test_prompts = [
    "Patient: I'm feeling overwhelmed with work and can't seem to manage my time.",
    "Patient: My anxiety is getting worse and I don't know what to do.",
    "Patient: I feel like I'm not good enough compared to everyone else.",
    "Patient: My relationship is falling apart and I'm scared."
]

for prompt in test_prompts:
    print(f"\n輸入: {prompt}")
    try:
        cop, response = generate_improved_response(model, tokenizer, prompt)
        print(f"\nCoP 分析:\n{cop}")
        print(f"\n治療師回應:\n{response}")
    except Exception as e:
        print(f"生成錯誤: {e}")
    print("-" * 70)

print("\n=== 訓練摘要 ===")
print(f"模型: {model_name}")
print(f"訓練樣本數: {len(train_dataset)}")
print(f"總訓練步數: ~{total_steps}")
print(f"LoRA rank: 32")
print("\n關鍵改進:")
print("1. 使用 [Therapeutic Response] 標記明確分隔")
print("2. 增加生成長度限制 (min_new_tokens=300)")
print("3. 實作重試機制確保生成品質")
print("4. 系統性構建包含6個元素的完整回應")
print("5. 備用回應機制避免空輸出")