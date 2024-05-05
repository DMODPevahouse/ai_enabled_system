# For Email Reinforcement Learning 
The purpose of this docker container is to take a json file, example given in the container, of data 
that consists of a companies userbase that emails are sent to, and the specific emails with determined SubjectLine_IDs
that the email was clicked on. The agent was trained on trying to determine what the subjectline of the email produced
the most reponses based on the users information, the state, and the action, the subjectline_id. Below are the commands
to run to get this information and to send in an email and their information to get a recommended subject line to send
to that user. 
### docker run --restart=unless-stopped -it -p 8793:8793 -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment14_1
There are a few commands that can be run after this image is started. 
### http://localhost:8793/predict -- allows you to send a json file to get an inference instead of the infer method above, an example json file is in /workspace, feel free to adjust amounts but stay within the same datatype and it will give a rating based off of the sentiment of the review
### http://localhost:8793/stats -- Plays the game to determine the rewards achieved by the agent, the agent will play the game and the stats will be returned
### http://localhost:8793/train -- trains the agent on the entire dataset, which may take an incredibly long time to run. At the time of writing this, it was tested for 8 hours and still did not finish.