# Further updated structure diagram incorporating the relationships between db, api, and agent as requested

final_structure_diagram = """
SandboxTwitterModule
|
|--- core
|    |
|    |-- twitter_controller.py
|    |   |-- class TwitterController (alias TCtrl)
|
|--- infra
|    |
|    |-- twitter_hci.py
|    |   |-- class TwitterHCI (alias THci)
|    |
|    |-- agents_graph.py
|    |   |-- class homoAgentsGraph
|    |       |-- uses AgentsGenerator (from agents_generator.py) ------->
|    |       |-- init TwitterUserAgent (alias tuAgent)                  v
|    |                                                                  |
|    |-- twitter_infra.py                                               v
|    |   |-- class TwitterInfra (alias TInfra)                          |
|    |                                                                  v
|    |-- agents_generator.py -------<----------<--------<-------<<------<
|    |   |-- class AgentsGenerator (used by homoAgentsGraph)
|    |
|    |-- Agent
|    |   |-- twitterUserAgent.py
|    |       |-- class TwitterUserAgent (alias tuAgent)
|    |          |-- uses TwitterAPI (from ../twitter_api.py) -------------->
|    |              |-- uses TwitterDB (from ../twitter_db.py)------->     v
|    |                                                               v     |
|    |-- twitter_db.py                                               |     v
|    |   |-- class TwitterDB -----<-----------<------------<---------<     |
|    |                                                                     v
|    |-- twitter_api.py                                                    |
|        |-- class TwitterApi ----<-----------<------------<---------------<
|
|--- core
|    |-- decorators.py
|    |   |-- function_call_logger (decorator)
|
|--- sandbox_twitter.py
     |-- class SandboxTwitter
         |-- __init__()
         |-- update_signal()
         |-- update_agents_db()
         |-- run()
         |-- RunSimula()
         |-- run_agent_lifecycle()
"""

print(final_structure_diagram)
