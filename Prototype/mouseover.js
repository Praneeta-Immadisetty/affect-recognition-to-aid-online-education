// JavaScript source code


//const x = document.getElementById("sec-abcf");
//x.addEventListener("MouseOver", RespondMouseOver);
function RespondMouseOver() {
    document.getElementById("line").innerHTML =
        "0-Frustrated"+"<br>"+"1-Distracted"+"<br>"+"2-Confused"+"<br>"+"3-Interested"+"<br>"+"4-Neutral"+"<br>"+"5-Bored"+"<br>"+"6-Surprised"+"<br>";
}

function RespondMouseOverPie() {
    fetch('D:/interships_projects/affect_recognition/Emotion_Detection_CNN-main/Model/p_analysis.txt')
    .then((res)=>{
        return res.text();
    }).then((data)=>{
        document.getElementById("pie").innerHTML =
            data + "<br>";
    })
}
