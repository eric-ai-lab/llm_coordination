Code repository for the paper "Evaluating Multi-Agent Coordination Abilities in Large Language Models" (https://arxiv.org/abs/2310.03903)

Abstract: A pivotal aim in contemporary AI research is to develop agents proficient in multi-agent coordination, enabling effective collaboration with both humans and other systems. Large Language Models (LLMs), with their notable ability to understand, generate, and interpret language in a human-like manner, stand out as promising candidates for the development of such agents. In this study, we build and assess the effectiveness of agents crafted using LLMs in various coordination scenarios. We introduce the LLM-Coordination (LLM-Co) Framework, specifically designed to enable LLMs to play coordination games. With the LLM-Co framework, we conduct our evaluation with three game environments and organize the evaluation into five aspects: Theory of Mind, Situated Reasoning, Sustained Coordination, Robustness to Partners, and Explicit Assistance. First, the evaluation of the Theory of Mind and Situated Reasoning reveals the capabilities of LLM to infer the partner's intention and reason actions accordingly. Then, the evaluation around Sustained Coordination and Robustness to Partners further showcases the ability of LLMs to coordinate with an unknown partner in complex long-horizon tasks, outperforming Reinforcement Learning baselines. Lastly, to test Explicit Assistance, which refers to the ability of an agent to offer help proactively, we introduce two novel layouts into the Overcooked-AI benchmark, examining if agents can prioritize helping their partners, sacrificing time that could have been spent on their tasks. This research underscores the promising capabilities of LLMs in sophisticated coordination environments and reveals the potential of LLMs in building strong real-world agents for multi-agent coordination.


The Overcooked environment used in this environment is based on https://github.com/HumanCompatibleAI/overcooked_ai/tree/master.

## Installation Instructions 
```
conda create -n llm_coordination python=3.7
conda activate llm_coordination
```

Clone the repository 
```
git clone https://github.com/eric-ai-lab/llm_coordination.git
```

Build from source
```
pip install -e llm_coordination/
```

Install requirements for Overcooked Demo 
```
pip install -r requirements.txt 
```

Usage: Visualize Multi-Agent Gameplay

```
cd src/overcooked_demo/server
python app.py
```
By default the server will run on localhost on port 5003. The default selected model will be gpt-3.5-turbo. Set openai API key and organization using Environment Variables API_KEY="Your_Key" and ORGANIZATION="Your_Organization.


## Citation
```
@misc{agashe2023evaluating,
      title={Evaluating Multi-Agent Coordination Abilities in Large Language Models}, 
      author={Saaket Agashe and Yue Fan and Xin Eric Wang},
      year={2023},
      eprint={2310.03903},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}   
```