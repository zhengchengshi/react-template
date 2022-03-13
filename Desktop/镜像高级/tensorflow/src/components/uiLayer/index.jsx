import React from 'react'
import * as PIXI from "pixi.js";
export default function UiLayer() {
    // 建立连接
    let type = "WebGL"
    if(!PIXI.utils.isWebGLSupported()){
      type = "canvas"
    }
    PIXI.utils.sayHello(type)

    // 创建PIXI应用
    let app = new PIXI.Application({width: 750, height: 256});

    //Add the canvas that Pixi automatically created for you to the HTML document
    document.body.appendChild(app.view);
    return (
        <div>UiLayer</div>
    )
}
