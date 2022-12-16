from flask import Flask, render_template, request,redirect,url_for
import os
import morph as morph
app = Flask(__name__) 

app.config['IMAGE_UPLOADS'] = 'C:/Users/preth/OneDrive/Desktop/umbc course works/Second semester/DAA/trail2'
from werkzeug.utils import secure_filename
@app.route('/') 
def student(): 
   return render_template('home.html') 
 
@app.route('/result',methods = ['POST', 'GET']) 
def result():
    if request.method == 'POST':
        image = request.files['filename']
        image1 = request.files['filename1']
        if image.filename == '' or image1.filename == '':
            print("File name is invalid")
            return redirect(request.url)
        filename = secure_filename(image.filename)
        filename1 = secure_filename(image1.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
        morph.Morph(filename,filename1)
        fname = 'output_video.mp4'
        return render_template("home.html",filename=fname)
    return render_template("home.html") 

@app.route('/display/<filename>')
def display_video(filename):
   return redirect(url_for('static',filename =filename),code=301)

if __name__ == '__main__': 
   app.run(debug = True) 