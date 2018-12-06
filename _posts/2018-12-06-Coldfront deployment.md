---
layout: single
author_profile: false
---

install LDAP dependences
```bash
yum install python-devel openldap-devel -y
pip install django-auth-ldap
pip install python-ldap ldap3
```

Change coldfront/config/local_settings.py
```bash
import ldap
from django_auth_ldap.config import GroupOfNamesType, LDAPSearch

AUTH_LDAP_START_TLS = False
AUTH_LDAP_BIND_AS_AUTHENTICATING_USER=True

AUTH_LDAP_SERVER_URI = ''
AUTH_LDAP_USER_DN_TEMPLATE = "uid=%(user)s,ou=People,dc=rcf,dc=unl,dc=edu"
AUTH_LDAP_USER_ATTR_MAP = {
    'username': 'uid',
    'first_name': 'givenName',
    'last_name': 'sn',
    'email': 'mail',
}

#AUTH_LDAP_SERVER_URI = "ldap://hcc-ldap01.unl.edu/"
#AUTH_LDAP_USER_SEARCH_BASE = 'dc=rcf,dc=unl,dc=edu'
#AUTH_LDAP_BIND_DN = "uid=%username%,ou=People,dc=rcf,dc=unl,dc=edu"
#AUTH_LDAP_BIND_PASSWORD = "phlebotinum"
#AUTH_LDAP_USER_SEARCH = LDAPSearch(
#    AUTH_LDAP_USER_SEARCH_BASE, ldap.SCOPE_ONELEVEL, '(uid=%(user)s')

#AUTH_LDAP_MIRROR_GROUPS = True
#AUTH_LDAP_GROUP_SEARCH_BASE = 'cn=groups,cn=accounts,dc=localhost,dc=localdomain'
#AUTH_LDAP_GROUP_SEARCH = LDAPSearch(
#    AUTH_LDAP_GROUP_SEARCH_BASE, ldap.SCOPE_ONELEVEL, '(objectClass=groupOfNames)')
#AUTH_LDAP_GROUP_TYPE = GroupOfNamesType()


EXTRA_AUTHENTICATION_BACKENDS += ['django_auth_ldap.backend.LDAPBackend',]
ADDITIONAL_USER_SEARCH_CLASSES = ['coldfront.plugins.ldap_user_search.utils.LDAPUserSearch',]
```
