from q_model import find_best_match,get_answer,catselector,answer_lines,lines,get_answer_lines,get_lines,idx


tex=input('What is your branch?')
answer_lines=get_answer_lines(tex)
lines=get_lines(tex)
print('Ask a question')

def reply(tex):
    num=find_best_match(tex, lines)
    idx=num
    catselector(num)
    return get_answer(idx,answer_lines)


    


""""
This works perfectly for all cases of faqs except for where we ave similar words in the questions.
We can solve that using  word2vec instead
"""