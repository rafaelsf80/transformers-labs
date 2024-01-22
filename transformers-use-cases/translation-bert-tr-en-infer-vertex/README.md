Dockerfile to run a prediction
 https://github.com/entrpn/serving-diffusion

I think all you would need to do is use the FROM cuda image and install transformers as a dependency and use the main.py inside the app folder and change it accordingly to your usecase
