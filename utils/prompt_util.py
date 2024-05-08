import glob
import json


class PromptEngine:
    def __init__(self, prompt_dir):
        self.prompt_dir = prompt_dir
        self.prompt_files = self._load_prompt_files(prompt_dir)
        self.prompt_templates = self._load_prompt_()

    def _load_prompt_files(self, prompt_dir):
        return sorted(glob.glob(prompt_dir + "/*.jsonl"))

    def _load_prompt_(self):
        prompt_templates = []
        for prompt_file_name in self.prompt_files:
            print(prompt_file_name)
            with open(prompt_file_name, 'r', encoding='utf-8') as fin:
                data = [json.loads(line) for line in fin.readlines()]
                prompt_templates.append(data)
        return prompt_templates

    def get_prompt(self, step):
        return self.prompt_templates[step][-1]

    def update_prompt(self, step, new_prompt):
        self.prompt_templates[step].append({"template": self.prompt_templates[step][0]['template'],
                                            "standard": new_prompt})
        with open(self.prompt_files[step], 'a', encoding='utf-8') as fout:
            fout.write(json.dumps({"template": self.prompt_templates[step][0]['template'],
                                   "standard": new_prompt}) + "\n")

    def check_stop(self):
        exp_flag = False
        rerank_flag = False
        if len(self.prompt_templates[0]) >= 3:
            if self.prompt_templates[0][-1]['standard'] == self.prompt_templates[0][-2]['standard'] == self.prompt_templates[0][-3]['standard']:
                exp_flag = True
        if len(self.prompt_templates[2]) >= 3:
            if self.prompt_templates[2][-1]['standard'] == self.prompt_templates[2][-2]['standard'] == self.prompt_templates[2][-3]['standard']:
                rerank_flag = True
        return exp_flag & rerank_flag


if __name__ == "__main__":
    prompt_engine = PromptEngine("../prompt_templates")
    print(prompt_engine.get_prompt(1))
    # prompt_engine.update_prompt(7, 'New Prompt')
    print(prompt_engine.check_stop())
