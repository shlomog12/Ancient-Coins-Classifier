const server_url = "http://localhost:3555/api/run_model";
const input = document.getElementById("inputTag");

let inputImage = null;

input.addEventListener("change", ()=>{
    const imageName = document.getElementById("imageName");
    inputImage = document.querySelector("input[type=file]").files[0];
    imageName.innerText = inputImage.name;
    console.log(inputImage.name);
})

async function runModel(){
  const imageName = document.getElementById("imageName");
  if (inputImage === null){
    imageName.innerText = "לא נבחרה תמונה"
    return;
  }
  clearOldResult();
  const res = await sendPhoto();
  showResult(res);
}

async function sendPhoto() {
    let formData = new FormData();
    formData.append("static_file", inputImage);
    const res = await fetch(server_url, {
      method: "POST",
      body: formData,   
    });
    const resJson = await res.json();
    return resJson.resModel;
}

function showResult(res){
  // transRes = {0: 'אלכסנדר ינאי', 1: 'יוחנן הורקנוס הראשון', 2: 'יהודה'};
  let resName;
  let resLink;
  let resImageName;
  switch (res) {
    case 0:
      resName = 'אלכסנדר ינאי';
      resLink = 'https://he.wikipedia.org/wiki/%D7%90%D7%9C%D7%9B%D7%A1%D7%A0%D7%93%D7%A8_%D7%99%D7%A0%D7%90%D7%99';
      resImageName = 'ALEXANDR.jpg';
      break;
    case 1:
      resName = 'יוחנן הורקנוס הראשון';
      resLink = 'https://he.wikipedia.org/wiki/%D7%99%D7%95%D7%97%D7%A0%D7%9F_%D7%94%D7%95%D7%A8%D7%A7%D7%A0%D7%95%D7%A1_%D7%94%D7%A8%D7%90%D7%A9%D7%95%D7%9F';
      resImageName = 'Hyrcanus-Yohanan.jpg';
      break;
    case 2:
      resName = 'יהודה';
      resLink = 'https://he.wikipedia.org/wiki/%D7%99%D7%94%D7%95%D7%93%D7%94_%D7%90%D7%A8%D7%99%D7%A1%D7%98%D7%95%D7%91%D7%95%D7%9C%D7%95%D7%A1_%D7%94%D7%A8%D7%90%D7%A9%D7%95%D7%9F';
      resImageName = 'Aristobulus-I.jpg';
      break;
  }

  const linkRes = document.getElementById("textRes");
  linkRes.textContent = resName;
  linkRes.href = resLink;
  const imgElement = document.getElementById('imageRes');
  imgElement.src = `images/${resImageName}`;
  imgElement.style.display = 'block';
}

function clearOldResult(){
  const imgElement = document.getElementById('imageRes');
  imgElement.style.display = 'none';
  const linkRes = document.getElementById("textRes");
  linkRes.textContent = '';
  linkRes.href = '';
}