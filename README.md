# ExperimentsWithAI
Simple project to experiment with various AI Python libraries.
Using some basic Python libraries to investigate the basics of NLP

Using the Python [NLTK library](https://www.nltk.org/) - Natural Language Toolkit

### Comparison of Porter Stemmer, Snowball Stemmer and WordNet Lemmatizer

| Word | Porter Stemmer | Snowball Stemmer | WordNet Lemmatizer | Noun             | Verb             | Adjective           | Adverb            | Adjective Satellite |
|------|----------------|-------------------|-------------------| -----------------|------------------|---------------------|-------------------|---------------------|
| connect | connect | connect | connect | connect | connect | connect | connect | connect |
| connected | connect | connect | connected | connected | connect | connected | connected | connected |
| connecting | connect | connect | connecting | connecting | connect | connecting | connecting | connecting |
| connection | connect | connect | connection | connection | connection | connection | connection | connection |
| connections | connect | connect | connection | connection | connections | connections | connections | connections |
| connectivity | connect | connect | connectivity | connectivity | connectivity | connectivity | connectivity | connectivity |
| connects | connect | connect | connects | connects | connect | connects | connects | connects |
| operate | oper | oper | operate | operate | operate | operate | operate | operate |
| operated | oper | oper | operated | operated | operate | operated | operated | operated |
| operates | oper | oper | operates | operates | operate | operates | operates | operates |
| operating | oper | oper | operating | operating | operate | operating | operating | operating |
| operation | oper | oper | operation | operation | operation | operation | operation | operation |
| operations | oper | oper | operation | operation | operations | operations | operations | operations |
| relate | relat | relat | relate | relate | relate | relate | relate | relate |
| related | relat | relat | related | related | relate | related | related | related |
| relates | relat | relat | relates | relates | relate | relates | relates | relates |
| relating | relat | relat | relating | relating | relate | relating | relating | relating |
| relation | relat | relat | relation | relation | relation | relation | relation | relation |
| relations | relat | relat | relation | relation | relations | relations | relations | relations |
| create | creat | creat | create | create | create | create | create | create |
| created | creat | creat | created | created | create | created | created | created |
| creates | creat | creat | creates | creates | create | creates | creates | creates |
| creating | creat | creat | creating | creating | create | creating | creating | creating |
| creation | creation | creation | creation | creation | creation | creation | creation | creation |
| creations | creation | creation | creation | creation | creations | creations | creations | creations |
| analyze | analyz | analyz | analyze | analyze | analyze | analyze | analyze | analyze |
| analyzed | analyz | analyz | analyzed | analyzed | analyze | analyzed | analyzed | analyzed |
| analyzes | analyz | analyz | analyzes | analyzes | analyze | analyzes | analyzes | analyzes |
| analyzing | analyz | analyz | analyzing | analyzing | analyze | analyzing | analyzing | analyzing |
| analysis | analysi | analysi | analysis | analysis | analysis | analysis | analysis | analysis |
| analyses | analys | analys | analysis | analysis | analyse | analyses | analyses | analyses |
| learned | learn | learn | learned | learned | learn | learned | learned | learned |
| learning | learn | learn | learning | learning | learn | learning | learning | learning |
| learn | learn | learn | learn | learn | learn | learn | learn | learn |
| learns | learn | learn | learns | learns | learn | learns | learns | learns |
| learner | learner | learner | learner | learner | learner | learner | learner | learner |
| learners | learner | learner | learner | learner | learners | learners | learners | learners |
| run | run | run | run | run | run | run | run | run |
| running | run | run | running | running | run | running | running | running |
| runner | runner | runner | runner | runner | runner | runner | runner | runner |
| ran | ran | ran | ran | ran | run | ran | ran | ran |
| fly | fli | fli | fly | fly | fly | fly | fly | fly |
| flying | fli | fli | flying | flying | fly | flying | flying | flying |
| flew | flew | flew | flew | flew | fly | flew | flew | flew |
| flies | fli | fli | fly | fly | fly | flies | flies | flies |
| study | studi | studi | study | study | study | study | study | study |
| studies | studi | studi | study | study | study | studies | studies | studies |
| studying | studi | studi | studying | studying | study | studying | studying | studying |
| studied | studi | studi | studied | studied | study | studied | studied | studied |
| happy | happi | happi | happy | happy | happy | happy | happy | happy |
| happier | happier | happier | happier | happier | happier | happy | happier | happy |
| happiest | happiest | happiest | happiest | happiest | happiest | happy | happiest | happy |
| happiness | happi | happi | happiness | happiness | happiness | happiness | happiness | happiness |
| use | use | use | use | use | use | use | use | use |
| used | use | use | used | used | use | used | used | used |
| uses | use | use | us | us | use | uses | uses | uses |
| using | use | use | using | using | use | using | using | using |
| useful | use | use | useful | useful | useful | useful | useful | useful |
| useless | useless | useless | useless | useless | useless | useless | useless | useless |
| likes | like | like | like | like | like | likes | likes | likes |
| better | better | better | better | better | better | good | well | good |
| worse | wors | wors | worse | worse | worse | bad | worse | bad |
| running | run | run | running | running | run | running | running | running |
| ate | ate | ate | ate | ate | eat | ate | ate | ate |
| singing | sing | sing | singing | singing | sing | singing | singing | singing |
| wrote | wrote | wrote | wrote | wrote | write | wrote | wrote | wrote |
| driving | drive | drive | driving | driving | drive | driving | driving | driving |
| flying | fli | fli | flying | flying | fly | flying | flying | flying |
| went | went | went | went | went | go | went | went | went |
| swimming | swim | swim | swimming | swimming | swim | swimming | swimming | swimming |
| spoke | spoke | spoke | spoke | spoke | speak | spoke | spoke | spoke |
| buying | buy | buy | buying | buying | buy | buying | buying | buying |
| geese | gees | gees | goose | goose | geese | geese | geese | geese |
| mice | mice | mice | mouse | mouse | mice | mice | mice | mice |
| children | children | children | child | child | children | children | children | children |
| feet | feet | feet | foot | foot | feet | feet | feet | feet |
| teeth | teeth | teeth | teeth | teeth | teeth | teeth | teeth | teeth |
| men | men | men | men | men | men | men | men | men |
| women | women | women | woman | woman | women | women | women | women |
| oxen | oxen | oxen | ox | ox | oxen | oxen | oxen | oxen |
| leaves | leav | leav | leaf | leaf | leave | leaves | leaves | leaves |
| data | data | data | data | data | data | data | data | data |
| better | better | better | better | better | better | good | well | good |
| worse | wors | wors | worse | worse | worse | bad | worse | bad |
| faster | faster | faster | faster | faster | faster | fast | faster | fast |
| happier | happier | happier | happier | happier | happier | happy | happier | happy |
| bigger | bigger | bigger | bigger | bigger | bigger | big | bigger | big |