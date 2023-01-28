import React from 'react';
import { useState} from 'react';
import './FormStyling.css'


function Form (props) {
    const [review,setReview] = useState(''); 
    console.log(review,"review")
     const[prediction,setPrediction]=useState(0)
     const[probability,setprobability]=useState(0)
     console.log(prediction,"prediction");
     
    // send request
    const request=(review) => {
    const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review })
    };
    fetch('http://127.0.0.1:8000/api/predict', requestOptions)
        .then(response => response.json())
        .then(data =>  {setPrediction(data.predictedrating)
        setprobability(data.probability)}
     
         );
   


    
 }
  
    const handleSubmit = (event) => {
        if(review.trim().length !== 0){
        event.preventDefault();
        console.log(review);
        }else{
            alert(`enter a valid text`)
        }
      }
     

      return( 
       <div className='background' >
        <h1 className='title'> Predicting Amazon Reviews</h1>
        <form className='form' onSubmit={handleSubmit}>
        <label>
        <input placeholder="Type something…"  className='input'   onChange={(e) => setReview(e.target.value)} type="text" />
        </label>
        <button onClick={e=>request(review)}  className="button" type="submit">Predict</button>
        </form>
        <h1  className='probability'><label>{"Probability"}</label><label className='probability1'>{probability}</label></h1>

        <h1>{props.getData(prediction)}</h1>
        </div>
     );   
}



export default Form;


