"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from zim.utils import AverageMeter, print_once
from .metric import compute_eval_scores, get_gradfilter

def run_eval(
    config, 
    valloader,
    evaluator_dict
):
    score_dict = {}
    
    for model_name, evaluator in evaluator_dict.items():
        print_once(f'\nLOG) {model_name} evaluation start.')
        model_score_dict = {}
        for name, loader in valloader.items():
            print_once(f"LOG) evaluate {model_name} on {name} dataset")

            score = evaluate(
                name=name,
                evaluator=evaluator,
                dataloader=loader,
                prompt_type=config.eval.prompt_type,
                use_ddp=config.use_ddp,
                enable_amp=config.use_amp,
                model_name=model_name,
            )
            model_score_dict[name] = score
        score_dict[model_name] = model_score_dict
        
    print_once(f'\nLOG)All evaluation done. Result : ')

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        result = "\n============================================\n"
        
        for k, v in score_dict.items():
            print_once('\n')
            for data_type, log in v.items():
                for _k, _v in log.items():
                    for __k, __v in _v.items():
                        result += f'Model: {k}'
                        result += f', Prompt: {_k}'
                        result += f', Scale: {__k}'
                        result += f', dataset: {config.dataset.valset} ({data_type})\n'
                        result += f'{__v}\n\n'
        result += "============================================\n"
        print_once(result)

        
def evaluate(
    name,
    evaluator,
    dataloader,
    prompt_type,
    use_ddp,
    enable_amp,
    model_name,
):

    metric_list = ["l1", "l2", "grad", "conn", "sad"]
    
    scale_list = ["all", "S", "M", "L"]
    #scale_list = ["all", ]
    
    average_metric = {
        prompt: {
            scale: {
                metric_name: AverageMeter(use_ddp) for metric_name in metric_list} 
            for scale in scale_list} 
        for prompt in prompt_type
    }
        
    batch_size = dataloader.batch_size
    device = evaluator.device
    grad_filter = get_gradfilter(device)

    for _iter, batched_input in enumerate(dataloader):
        for k, v in batched_input.items():
            if type(v) == torch.Tensor:
                batched_input[k] = v.to(device)

        with torch.cuda.amp.autocast(enabled=enable_amp) and torch.no_grad():
            batched_output = evaluator(batched_input)

        ratio = batched_input['ratio'][0]
    
        for prompt in prompt_type:
            logits = batched_output[prompt]["masks"]
            mattes = batched_input["mattes"]

            scores = compute_eval_scores(
                logits, mattes, grad_filter,
            )

            for m in metric_list:
                average_metric[prompt]["all"][m].update(scores[m], batch_size)
                
                if "S" in average_metric[prompt] and ratio < 0.01:
                    average_metric[prompt]["S"][m].update(scores[m], batch_size)
                elif "M" in average_metric[prompt] and ratio < 0.1:
                    average_metric[prompt]["M"][m].update(scores[m], batch_size)
                elif "L" in average_metric[prompt] and ratio >= 0.1:
                    average_metric[prompt]["L"][m].update(scores[m], batch_size)

    # gather the stats from all processes
    result_dict = {}
    for prompt in prompt_type:
        result_dict[prompt] = {}
        
        for scale in scale_list:
            for k, v in average_metric[prompt][scale].items():
                v.synch(device)

            res = "Result"
            res += f" | MSE {average_metric[prompt][scale]['l2'].avg:.4f}"
            res += f" | SAD {average_metric[prompt][scale]['sad'].avg:.4f}"
            res += f" | MAE {average_metric[prompt][scale]['l1'].avg:.4f}"
            res += f" | Grad {average_metric[prompt][scale]['grad'].avg:.4f}"
            res += f" | Conn {average_metric[prompt][scale]['conn'].avg:.4f}"
                
            result_dict[prompt][scale] = res

    return result_dict
