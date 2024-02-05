import speech_recognition as sr
import csv

def main():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        print("Please say something")

        audio = r.listen(source)

        print("Recognizing Now .... ")

        try:
            print(r.recognize_google(audio))
            print("Audio Recorded Successfully \n ")


        except Exception as e:
            print("Error :  " + str(e))

        with open("recorded.wav", "wb") as f:
            f.write(audio.get_wav_data())

    data = [r.recognize_google(audio)]
    file = open("recorded.csv",'a+',newline='')
    writer = csv.writer(file)
    writer.writerows(data)
    file.close()

    file1 = open("recorded1.txt",'w')
    file1.write(r.recognize_google(audio))
    file1.close()



if __name__ == "__main__":
    main()