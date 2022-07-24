const server_url = "http://localhost:3555/api/run_model";
let input = document.getElementById("inputTag");
let imageName = document.getElementById("imageName")

let inputImage = null;

input.addEventListener("change", ()=>{
    inputImage = document.querySelector("input[type=file]").files[0];
    imageName.innerText = inputImage.name;
    console.log(inputImage.name);
})


async function sendPhoto(event) {
    if (inputImage === null){
        imageName.innerText = "לא נבחרה תמונה"
        return;
    }
    let formData = new FormData();
    formData.append("static_file", inputImage);


    console.log(inputImage)

    const res = await fetch(server_url, {
      method: "POST",
      body: formData,   
    });
    const xxx = await res.json();
    console.log(res.status);
    document.getElementById("textRes").textContent = xxx.resModel;


  }