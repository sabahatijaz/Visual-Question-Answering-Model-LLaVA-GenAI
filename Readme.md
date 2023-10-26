commands to run the code:
1) pip install requirements.txt
2) first call load function from the app.py file. it will oad model controllers and workers on the defined ports from env file.
3) then call model_inference from app.py, pass the byte encoded image or image url, it will take model name and prompt from env and will get the inference.
4) tools.py is used for logging purpose.
