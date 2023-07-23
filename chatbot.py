from q_model import flag,find_best_match,get_answer,catselector,answer_lines,lines,get_answer_lines,get_lines,idx,pre_quest
from translation import isGuj,en_input,toGuj
from entogu import toEng
from general import chat

tex=input('What is your branch?')
answer_lines=get_answer_lines(tex)
lines=get_lines(tex)
print('Ask a question')

def reply(tex):
    guj=0
    if(isGuj(tex)):
        guj=1
    tex=en_input(tex)
    cat=pre_quest(tex)
    if cat==0:
        num=find_best_match(tex, lines)
        idx=num
        catselector(num)
        print(flag)
        if flag<3:
            if guj==1:
                return toEng(get_answer(idx,answer_lines))
            return get_answer(idx, answer_lines)
    if cat==1:
        pass
    if cat==2:
        if guj==1:
                return toEng(chat(tex))
        return chat(tex)
    # if flag==3:
    #     pass
    # else:
    #     chat(tex)
    


""""
This works perfectly for all cases of faqs except for where we ave similar words in the questions.
We can solve that using  word2vec instead
"""