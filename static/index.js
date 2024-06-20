(function(){
    const http = new XMLHttpRequest(); 
    var canvas = document.querySelector("#input");
    var ctx = canvas.getContext("2d");
    var mousedown = false; 
  
    const submitButton = document.querySelector(".submit")
    const clearButton = document.querySelector(".clear") 
    
    // get the canvas context 
    ctx.strokestyle = "black"; 
    ctx.lineWidth = 6; 
    ctx.lineJoin= "round"; 
    
    
    function draw_stroke(e){
      if(!mousedown){
        return 
      }
    
      var x = e.clientX - canvas.offsetLeft; 
      var y = e.clientY - canvas.offsetTop; 
    
      ctx.lineTo(x, y); 
      ctx.stroke(); 
      ctx.beginPath(); 
      ctx.moveTo(x,y); 
    
    }
    
    function submitQuery(){
      let predictionElement = document.querySelector("#class-pred"); 
      predictionElement.textContent = "Running model, please wait: ";

      const image = canvas.toDataURL();
    // // const image = canvas.toDataURL(type="image/jpeg");
    // const image = canvas.toDataURL("image/jpeg", 1.0);
      
      let url = "/predict"
    
      http.open("POST", url) 
    
      http.send(image) 
    
      http.onload = function(){
          
        if (http.status === 200){
          let predictionElement = document.querySelector("#class-pred"); 

          // let response_obj = http.response; 
          // let path = `{{url_for('static', filename=${response_obj})}}`; 

          // let img = document.getElementsByClassName("image")
          // // img.src = path;
          // img.src = "{{url_for('static', filename='output.jpg')}}";

          // window.alert(response_obj)
          // console.log(response_obj)
          // console.log(predictionElement)
          // console.log(path)
          location.reload();

          predictionElement.textContent = "Model output: ";
          // predictionElement.textContent = `predicted class from the model: ${path}`; 
        }else{
          predictionElement.textContent = "An error occured"; 
        }
         
      }
    
    }
    
    function clearCanvas(){
      ctx.clearRect(0, 0, canvas.width, canvas.height); 
      let predictionElement = document.querySelector("#class-pred"); 
      predictionElement.textContent = ""; 
    }
    
    // add the event listeners 
    canvas.addEventListener("mousedown", ()=>{mousedown = true}); 
    canvas.addEventListener("mousemove", draw_stroke); 
    canvas.addEventListener("mouseup", ()=>{ mousedown = false; ctx.beginPath()}); 
    submitButton.addEventListener("click", submitQuery); 
    clearButton.addEventListener("click", clearCanvas);
  })();