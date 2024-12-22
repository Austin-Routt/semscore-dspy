
# SemScore Validation Report
Generated on: 2024-12-22 01:31:25

## OpenAssistant Dataset Results

### TinyLlama/TinyLlama-1.1B-Chat-v1.0


Performance Metrics:
- Pearson Correlation: 1.0000
- Mean Absolute Difference: 0.0000
- Mean Calculated Score: 0.6974
- Mean Reference Score: 0.6974

Example Cases:
| Prompt | Prediction | Reference | Calculated | Expected | Diff |
|--------|------------|-----------|------------|-----------|------|
| <|user|> Can photoshop be used as a tool to forge ... | Yes, photoshop can be used as a tool to forge docu... | Yes, Photoshop can be used to forge documents due ... | 0.9537 | 0.9537 | 0.0000 |
| <|user|> What is the difference between C# and Jav... | Certainly! Here are some key differences between C... | Sure, I can provide you with a comparison between ... | 0.8042 | 0.8042 | 0.0000 |
| <|user|> Explain GPIO pins on a raspberry pi pico ... | GPIO (General Purpose Input/Output) pins are used ... | GPIO (General Purpose Input/Output) pins are physi... | 0.8529 | 0.8529 | 0.0000 |

### models/Mistral-7B-Instruct-v0.2


Performance Metrics:
- Pearson Correlation: 1.0000
- Mean Absolute Difference: 0.0000
- Mean Calculated Score: 0.7372
- Mean Reference Score: 0.7372

Example Cases:
| Prompt | Prediction | Reference | Calculated | Expected | Diff |
|--------|------------|-----------|------------|-----------|------|
| <s>[INST] Can photoshop be used as a tool to forge... | Yes, Adobe Photoshop and similar image editing sof... | Yes, Photoshop can be used to forge documents due ... | 0.8402 | 0.8402 | 0.0000 |
| <s>[INST] What is the difference between C# and Ja... | Both C# and Java are popular programming languages... | Sure, I can provide you with a comparison between ... | 0.9332 | 0.9332 | 0.0000 |
| <s>[INST] Explain GPIO pins on a raspberry pi pico... | The Raspberry Pi Pico W is a microcontroller board... | GPIO (General Purpose Input/Output) pins are physi... | 0.7874 | 0.7874 | 0.0000 |
