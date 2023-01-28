import React from 'react'
import Form from "./Form";
import{FaStar} from "react-icons/fa"
import './StarStyling.css'

function StarReviews(params) {
    const getData =(data)=>{
        if(data==0)
        {
            return <div div className='divstar'> <label className='model'>{"The Model Predicted :"}</label><FaStar/><FaStar/><FaStar/><FaStar/><FaStar/></div>
        }
        if (data==1)
        {
            console.log("testmodel");
             return <div className='divstar'><label className='model'>{"The Model Predicted :"}</label><label className='star'><FaStar/></label><FaStar/><FaStar/><FaStar/><FaStar/></div>
        } if(data==2)
        {
            console.log("chibegam");
            return <div className='divstar'><label className='model'>{"The Model Predicted :"}</label><label className='star'><FaStar/><FaStar/></label><FaStar/><FaStar/><FaStar/></div>
        }if(data==3)
        {
            return (<div className='divstar'><label className='model'>{"The Model Predicted :"}</label><label className='star'><FaStar/><FaStar/><FaStar/></label><FaStar/><FaStar/></div>)
        }if(data==4)
        {
            return (<div className='divstar'><label className='model'>{"The Model Predicted :"}</label><label className='star'><FaStar/><FaStar/><FaStar/><FaStar/></label><FaStar/></div>)
        }if(data==5)
        {
           return(<div className='divstar'><label className='model'>{"The Model Predicted :"}</label><label className='star'><FaStar/><FaStar/><FaStar/><FaStar/><FaStar/></label></div>)
        }

    }

 
    return (<Form getData={getData} />);
}


export default StarReviews; 