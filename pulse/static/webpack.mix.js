let mix=require('laravel-mix');
require('laravel-mix-polyfill');

mix
    .ts('./scripts/app.ts','../static/assets/scripts/pulse-app.js')
    .sass('./styles/main.scss','../static/assets/styles/pulse-style.css')
    .sourceMaps()
    .options({
        processCssUrls: false,
        autoprefixer: {remove: false},
        postCss: [
            require('postcss-css-variables')({
                preserve: true,
                preserveInjectedVariables: false,
            }),
        ]
    })
    .polyfill({
        enabled: true,
        useBuiltIns: "usage",
        targets: {"firefox": "50","ie": 11},
        corejs: 3,
    })
    .browserSync({
        proxy: 'https://pulse.test/pulse/templates/index.html',
        files: [
            '../../static/assets/scripts/*.js',
            '../../static/assets/styles/*.css',
            '../../templates/*.html',
        ],
        ghostMode: false,
    });


