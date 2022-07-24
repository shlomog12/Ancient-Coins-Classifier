const express = require('express');
const cors=require("cors");
const app = express();
const port = 3000;


const corsOptions ={
  origin:'*', 
  credentials:true,            //access-control-allow-credentials:true
  optionSuccessStatus:200,
}

app.use(cors(corsOptions));
app.use(express.static('public'));



app.listen(port, () => {
    console.log(`listening on port ${port}`);
  })