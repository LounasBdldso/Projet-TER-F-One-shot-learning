import { Brain, Circle, Heart, Search } from 'lucide-react'
import React from 'react'
import hero from '../assets/hero.jpg'

const Hero = () => {
  return (
    <section className='relative mt-9 overflow-hidden bg-gray-50 py-12 px-4
    sm:px-6 md:px-8 lg:px-20 flex flex-col lg:flex-row items-center 
    justify-between'>
        <div className='flex-1 lg:ml-28 w-full max-w-xl space-y-6 relative
        z-20'>
            <h2 className='text-4xl sm:text-5xl font-bold text-gray-900 text-center 
            lg:text-left '
            data-aos ="fade-right" 
            >
                Qui {" "}
                <span className='font-bold text-black block lg:inline' >
                    est qui 
                    <span className='text-blue-900'>?</span>

                </span>
            </h2>
            <div className='flex gap-3 mt-4 justify-center lg:justify-start'
            data-aos ="fade-right"
            data-aos-delay="100">
                <Circle className='text-blue-500 w-5 h-5'></Circle>
                <Circle className='text-blue-700 w-5 h-5'></Circle>
                <Circle className='text-blue-900 w-5 h-5'></Circle>
            </div>
            <div className='flex flex-col lg:grid lg:grid-cols-2 gap-6 lg:gap-48
            items-center lg:items-start '>
                <div 
                data-aos='zoom-in' 
                data-aos-delay="200"
                className='bg-white w-72 rounded-3xl p-6 border  
                border-gray-100 shadow-xl lg:shadow-2xl transition '>
                    <div className='ml-4 w-10 h-10 flex items-center justify-center
                    rounded-lg bg-blue-100 mb-4 '>
                        <span className='text-blue-500 text-xl '>
                            <Brain className='text-blue-600 w-6 h-6 '/>
                        </span>
                    </div>
                    <h3 className='ml-4 text-lg font-semibold text-gray-800 '> 
                        Comment cela marche ?

                    </h3>
                    <p className='ml-4 text-gray-500 text-sm mt-2'>
                        Lorem ipsum dolor sit amet, consectetur
                    </p>
                </div>

                <div 
                data-aos='zoom-in'
                data-aos-delay="300"
                className='bg-white w-72 rounded-3xl p-6 border  
                border-gray-100 shadow-xl lg:shadow-2xl transition '>
                    <div className='ml-4 w-10 h-10 flex items-center justify-center
                    rounded-lg bg-blue-100 mb-4 '>
                        <span className='text-blue-700 text-xl '>
                            <Search className='text-blue-600 w-6 h-6 '/>
                        </span>
                    </div>
                    <h3 className='ml-4 text-lg font-semibold text-gray-800 '> 
                        we only lookk once?

                    </h3>
                    <p className='ml-4 text-gray-500 text-sm mt-2'>
                        Lorem ipsum dolor sit amet, consectetur
                    </p>
                </div>

            </div>
        </div>
        <div 
        data-aos="fade-left"
        data-aos-delay="400"
        className='flex-1  mt-12 lg:mt-0 lg:ml-12 relative w-full flex 
        justify-center'>
            <div className='w-64 h-64 sm:w-80 sm:h-80 md:w-96 md:h-96 
            lg:w-[600px] lg:h-[500px] overflow-hidden border-8 border-white
            shadow-lg lg:shadow-lg
            rounded-[60%_40%_30%_60%/60%_30%_70%_30%] relative z-10'>
                <img 
                src={hero} 
                alt="tête numérique" 
                className='object-cover w-full h-full'/>

            </div>


        </div>

    </section>
  )
}

export default Hero