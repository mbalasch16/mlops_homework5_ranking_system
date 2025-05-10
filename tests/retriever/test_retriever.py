from src.retriever.retriever import get_similar_responses
import numpy as np

# def test_get_similar_responses():
#     assert get_similar_responses("What is the capital of France?") == ["These are test responses"]

def test_get_similar_responses():
    result = get_similar_responses("Famous movie in hindi")

    # Expected output 
    expected = [
         {
    "excerpt": "Ashutosh Gowariker: Lagaan was the story of a cricket match between British officers and Indian villagers in the late 18th century It ranked third among 2001's Indian movies in terms of gross revenue In 2004, Gowariker directed Swades, starring Shahrukh Khan Jodhaa Akbar (2008), another historical epic romance set in the 16th century, starred Hrithik Roshan and Aishwarya Rai As a filmmaker he shows his own civility through his films",
    "score": 0.5889000296592712
  },
  {
    "excerpt": "Sri Ramadasu: Upon release, the film got highly positive reviews The film's lead actor Nagarjuna received unanimous positive appreciation for his portrayal in the titular role and subsequently went on to win Nandi Award for Best Actor that year Music director M M Keeravani also received rave reviews for his work Along with being critically acclaimed,  The film recorded as Blockbuster hit at the box office",
    "score": 0.5418999791145325
  },
  {
    "excerpt": "Sunayana Fozdar: I D , Savdhaan India, Mahisagar, Aahat, and Yam Hain Hum on SAB TV, Ek Rishta Saajhedari Ka on Sony TV and Belan Wali Bahu on Colors  TV Sony TV's Meet Mila De Rabba,  C I D SAB TV's Hansi He Hansi   Mil Toh Lein",
    "score": 0.5394999980926514
  },
  {
    "excerpt": "Eradu Kanasu: K S Ashwath  B Jayashree The song Endendu Ninnanu was shot at Gajanur dam in Shivamoga district The film was based on the novel of the same name by Vani Muralidhara Khajane of The Hindu mentioned that the plot was similar to William Shakespeare's Romeo and Juliet Kalpana had a more prominent role than Dr Raj Kumar The music was composed by Rajan-Nagendra with lyrics for the soundtracks penned by Chi",
    "score": 0.5232999920845032
  },
  {
    "excerpt": "Bobbili Brahmanna: Later, the formula was used in Cheran Pandiyan (1991) (Tamil film, remade in Telugu as Balarama Krishnulu in 1992), Chinna Gounder (1992) (Tamil film, remade in Telugu as Chinarayudu in 1992), Bobbili Simham (1994), Nattamai (1994) (Tamil film, remade in Telugu as Pedarayudu in 1995), Kondapalli Rattaiah (1995) and Kondaveeti Simhasanam (2002) ",
    "score": 0.5169000029563904
  }
   
       
        
    ]

    # Assert that the result matches the expected output
    assert result == expected
    #assert result = [{'excerpt' : 'These are sample responses', score}]
   
#test to check returned values are a list
def test_get_similar_responses_returns_list():
    result = get_similar_responses("hello")
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)

#test to change the custom value of top_k values
def test_custom_top_k():
    top_k = 3
    result = get_similar_responses("Tell me about bollywood movie", top_k=top_k)
    assert len(result) <= top_k



