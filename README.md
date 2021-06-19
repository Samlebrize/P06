Tutoriel : 

This application purpose is to get the user an idea of the best tags to add to a new post in Stackoverflow, 

on the URL Romainb.pythonanywhere.com, fill the title of your post and the question in itself and click on submit button. 

It will give you a maximum number of 5 best tags to add to your post in order to maximize the chance to be answered because of a better search engine optimisation. 

The website is composed of 2 HTML files : a first one index.html that is the main page to request the title and the body of the topic. 

Those 2 variables are then processed through flask_app.py file that use the previously saved models to fit the data, then search for the best tags for this topic and get the output under list format. 

The list is then printed in the results.html file