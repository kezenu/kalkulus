from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("klasifikasi_jeruk.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    confidence = None

    if request.method == "POST":
        diameter = float(request.form["diameter"])
        berat = float(request.form["berat"])
        tebal_kulit = float(request.form["tebal_kulit"])
        kadar_gula = float(request.form["kadar_gula"])
        asal_daerah = request.form["asal_daerah"]
        warna = request.form["warna"]
        musim_panen = request.form["musim_panen"]

        data = pd.DataFrame([[
            diameter,
            berat,
            tebal_kulit,
            kadar_gula,
            asal_daerah,
            warna,
            musim_panen
        ]], columns=[
            'diameter',
            'berat',
            'tebal_kulit',
            'kadar_gula',
            'asal_daerah',
            'warna',
            'musim_panen'
        ])

        prediksi = model.predict(data)[0]
        persentase = max(model.predict_proba(data)[0])

        hasil = prediksi
        confidence = f"{persentase:.2%}"

    return render_template("index.html", hasil=hasil, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)