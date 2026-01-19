import json
import os.path

import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

def run_training_data(agent: ArcAgent, arc_problems: list[ArcProblem]) -> dict[ArcProblem, tuple[bool, list]]:
    """
    Run each training problem with the test output included so the agent can
    test if they are getting the correct response.
    """
    train_ans_dict: dict[ArcProblem, tuple[bool, list]] = dict()
    for trn_problem in arc_problems:
        preds: list[np.ndarray] = agent.make_predictions(trn_problem)
        correct = False

        if len(preds) <= 3:
            for prediction in preds:
                answer = trn_problem.test_set().get_output_data().data()
                correct = np.array_equal(answer, prediction)
                if correct: break

        # # store the problem_set and whether it was correctly solved
        train_ans_dict[trn_problem] = (correct, preds)

    return train_ans_dict

def load_arc_problems(path: str, problem_data: list[str]) -> list[ArcProblem]:
    problems: list[ArcProblem] = list()
    for problem_name in problem_data:
        with open(os.path.join(path, problem_name)) as p:
            flat_data: dict[str, dict] = json.load(p)
            # convert the data into ArcData (i.e. numpy.ndarray data)
            trn_data: list[ArcSet] = list()
            for dt in flat_data['train']:
                d_input = ArcData(np.array(dt['input']))
                d_output = ArcData(np.array(dt['output']))
                trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
                trn_data.append(trn_set)

            tst_data: list[ArcSet] = list()
            for tst in flat_data['test']:
                t_input = ArcData(np.array(tst['input']))
                t_output = ArcData(np.array(tst['output']))
                tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
                tst_data.append(tst_set)

            arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_data[0])

            # # there should only be one test in the test data
            problems.append(arc_problem)

    return problems


if __name__ == "__main__":
    
    # Here you can use this to open other milestone data directories for running against
    #  you'll should copy this code and change the path to the milestone you want to load (B, C or D)
    milestone_path = os.path.join('Milestones', 'B')
    milestone_data: list[str] = os.listdir(milestone_path)

    arc_milestone_problems: list[ArcProblem] = load_arc_problems(milestone_path, milestone_data)

    # instantiate the agent once
    arc_agent: ArcAgent = ArcAgent()

    milestone_data_set = run_training_data(arc_agent, arc_milestone_problems)
    milestone_file = open('Milestone_Results.csv', 'w')
    milestone_file.write("Problem Name, Correct, Correct Answer, Prediction 1, Prediction 2, Prediction 3\n")
    for m_answer_set in milestone_data_set.keys():
        m_correct, predictions = milestone_data_set[m_answer_set]
        m_cor_ans = m_answer_set.test_set().get_output_data().data().tolist()
        milestone_file.write(f'{m_answer_set.problem_name()},'
                             f'{m_correct},'
                             f'"{m_cor_ans}",')
        if len(predictions) == 0:
            milestone_file.write("empty\n")
            continue
        for idx, pred in enumerate(predictions, 1):
            if len(predictions) == idx:
                milestone_file.write(f'"{pred.tolist()}"\n')
            else:
                milestone_file.write(f'"{pred.tolist()}",')

    milestone_file.close()
