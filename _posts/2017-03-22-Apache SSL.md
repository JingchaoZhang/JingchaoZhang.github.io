---
layout: single
author_profile: false
---

xdmod.conf
```bash
# Redirect HTTP to HTTPS
<VirtualHost *:80>
   ServerName hcc-xdmod.unl.edu
   Redirect / https://hcc-xdmod.unl.edu
</VirtualHost>

# BEGIN https://cipherli.st/
SSLStaplingCache "shmcb:logs/stapling-cache(150000)"
# END https://cipherli.st/

<VirtualHost *:443>
    ServerName hcc-xdmod.unl.edu

    ## Customize this section using your SSL certificate.
    SSLEngine on
    SSLCertificateChainFile /etc/letsencrypt/live/hcc-xdmod.unl.edu/chain.pem
    SSLCertificateFile      /etc/letsencrypt/live/hcc-xdmod.unl.edu/fullchain.pem
    SSLCertificateKeyFile   /etc/letsencrypt/live/hcc-xdmod.unl.edu/privkey.pem
    <FilesMatch "\.(cgi|shtml|phtml|php)$">
        SSLOptions +StdEnvVars
    </FilesMatch>

    # BEGIN https://cipherli.st/
    SSLCipherSuite EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH
    SSLProtocol All -SSLv2 -SSLv3
    SSLHonorCipherOrder On
    Header always set Strict-Transport-Security "max-age=21600; includeSubDomains; preload"
    #Header always set X-Frame-Options DENY
    Header always set X-Frame-Options SAMEORIGIN
    Header always set X-Content-Type-Options nosniff
    SSLCompression off
    SSLUseStapling on
    # END https://cipherli.st/

    DocumentRoot /usr/share/xdmod/html

    <Directory /usr/share/xdmod/html>
        Options FollowSymLinks
        AllowOverride All
        DirectoryIndex index.php index.html

        # Apache 2.4 access controls.
        <IfModule mod_authz_core.c>
            Require all granted
        </IfModule>
    </Directory>

    <Directory /usr/share/xdmod/html/rest>
        RewriteEngine On
        RewriteRule (.*) index.php [L]
    </Directory>

    ## SimpleSAML federated authentication.
    SetEnv SIMPLESAMLPHP_CONFIG_DIR /etc/xdmod/simplesamlphp/config
    Alias /simplesaml /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/www
    <Directory /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/www>
        Options FollowSymLinks
        AllowOverride All
        # Apache 2.4 access controls.
        <IfModule mod_authz_core.c>
            Require all granted
        </IfModule>
    </Directory>

    ErrorLog /var/log/xdmod/apache-error.log
    CustomLog /var/log/xdmod/apache-access.log combined
</VirtualHost>
```

[Template](https://cipherli.st/)  
[SSL Check](https://ssldecoder.org/)
