import streamlit as st
import torch
import numpy as np
from torchvision import transforms 
from model import Resnet
from PIL import Image
import torch.nn.functional as F

def predict(image, labels, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    # 学習済みのモデルの読み込み
    model.eval()
    outputs = model(image)

    y_prob = F.softmax(outputs.squeeze(0),dim=-1)
    sorted_prob,sorted_indices = torch.sort(y_prob,descending=True)

    results = []
    for prob, idx in zip(sorted_prob, sorted_indices):
        results.append((labels[idx.item()], prob.item()))

    return results

    #model = net.load_state_dict(torch.load('./model_judgeFaceEmotion.pth',map_location=torch.device('cpu')))

def main():
    model = Resnet(num_classes=3)
    model.load_state_dict(torch.load('model_judgeFaceEmotion.pth',map_location='cpu'))

    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    st.sidebar.title("感情の分類アプリ")
    st.sidebar.write("画像認識モデルを使って感情を分類します。")
    st.sidebar.write("判別が可能な種類は以下の通りです。")

    img_source = st.sidebar.radio("画像のソースを選択してください",
                    ("画像をアップロード", "画像を撮影"))
    if img_source == "画像をアップロード":
        img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
    else:
        img_file = st.camera_input("カメラで撮影")    

    if img_file is not None:
        with st.spinner("計算中・・・"):
            img = Image.open(img_file)
            st.image(img, caption="対象画像")
            st.write("")

            results = predict(img, labels, model)

            st.subheader("判定結果")
            num_top = 5
            for result in results[:num_top]:
                st.write(str(round(result[1] * 100, 2)) + "%の確率で" + result[0] + "です。")

if __name__ == "__main__":
    main()